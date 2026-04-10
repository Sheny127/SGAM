import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm

# =================================================================
# 1. 核心算法工具 (已修复稀疏矩阵计算)
# =================================================================

def pairwise_dist(pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height):
    device = pixel_features.device
    b, c, h, w = pixel_features.shape
    num_spixels = spixel_features.shape[-1]
    
    pixel_features = pixel_features.reshape(b, c, -1).permute(0, 2, 1) 
    spixel_features = spixel_features.permute(0, 2, 1) # (B, M, C)
    init_label_map = init_label_map.reshape(b, -1)
    
    dy = torch.tensor([-1, 0, 1], device=device)
    dx = torch.tensor([-1, 0, 1], device=device)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
    
    offsets = grid_y * num_spixels_width + grid_x
    offsets = offsets.reshape(-1)
    
    neighbor_indices = init_label_map.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
    neighbor_indices = neighbor_indices.clamp(0, num_spixels - 1).long()
    
    batch_indices = torch.arange(b, device=device).reshape(b, 1, 1).expand(b, h*w, 9)
    neighbor_features = spixel_features[batch_indices, neighbor_indices, :]
    
    pixel_features_expand = pixel_features.unsqueeze(2)
    dist = torch.sum((pixel_features_expand - neighbor_features) ** 2, dim=-1)
    return dist

class PairwiseDistFunction:
    @staticmethod
    def apply(pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height):
        return pairwise_dist(pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    batchsize, channels, height, width = images.shape
    device = images.device
    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))
    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)
    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)
    return centroids, init_label_map

@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()
    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)

@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()

@torch.no_grad()
def sparse_ssn_iter(pixel_features, num_spixels, n_iter):
    height, width = pixel_features.shape[-2:]
    batch_size = pixel_features.shape[0]
    
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(num_spixels / num_spixels_width)
    actual_nspix = num_spixels_width * num_spixels_height # 重新计算实际的 spix 数

    spixel_features, init_label_map = calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features_4d = pixel_features
    # shape (B, C, N) -> (B, N, C)
    pixel_features_flat = pixel_features.reshape(*pixel_features.shape[:2], -1).permute(0, 2, 1)

    for _ in range(n_iter):
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features_4d, 
            spixel_features, 
            init_label_map.view(batch_size, height, width), 
            num_spixels_width, num_spixels_height)

        affinity_matrix = (-dist_matrix).softmax(1)
        
        # === 修复的更新逻辑 START ===
        # 1. 构建稀疏矩阵 indices 和 values
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)
        mask = (abs_indices[1] >= 0) * (abs_indices[1] < actual_nspix)
        
        # 只取 mask 为 True 的部分
        valid_indices = abs_indices[:, mask]
        valid_values = reshaped_affinity_matrix[mask]
        
        # 2. 处理 Batch=1 的情况 (降维处理以支持 torch.sparse.mm)
        if batch_size == 1:
            # 去掉 batch 维度， indices 变成 (2, NNZ) -> [spix_idx, pix_idx]
            sp_indices = valid_indices[1:3, :] 
            sp_values = valid_values
            # 构造稀疏矩阵 (M, N)
            sparse_abs_affinity = torch.sparse_coo_tensor(
                sp_indices, sp_values, size=(actual_nspix, height*width)
            )
            
            # pixel features (1, N, C) -> (N, C)
            feat_dense = pixel_features_flat[0] 
            
            # 3. 稀疏矩阵乘法 (M, N) @ (N, C) -> (M, C)
            new_spixel_feat = torch.sparse.mm(sparse_abs_affinity, feat_dense)
            
            # 归一化因子
            normalizer = torch.sparse.sum(sparse_abs_affinity, 1).to_dense().unsqueeze(-1) + 1e-16
            new_spixel_feat = new_spixel_feat / normalizer
            
            # 还原维度 (M, C) -> (1, C, M) -> (1, M, C) ? 
            # 注意: spixel_features 期望是 (B, C, M)
            # new_spixel_feat 是 (M, C)
            spixel_features = new_spixel_feat.t().unsqueeze(0) # (1, C, M)
            
        else:
            # 如果 B > 1，由于 pytorch 对 batch sparse mm 支持不好，我们直接跳过更新
            # 保持 spixel_features 不变，只做 assignment
            pass
        # === 修复的更新逻辑 END ===

    hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)
    return None, hard_labels, None

# =================================================================
# 2. SSNModel 类
# =================================================================

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class SSNModel(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            conv_bn_relu(5, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64*3+5, feature_dim-5, 3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        pixel_f = self.feature_extract(x)
        return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)

    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)

# =================================================================
# 3. 主程序
# =================================================================

CONFIG = {
    'fdim': 20,           
    'nspix': 100,         
    'niter': 10,          
    'color_scale': 0.26,  
    'pos_scale': 2.5,     
    'model_path': 'best_model.pth', 
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

INPUT_DIR = r'D:\Lab\UAVVasteDataset\images'
OUTPUT_DIR = 'datasets/UAVVaste/ssn/train'
INFER_SIZE = (960, 540) 

def load_model():
    print(f"Loading model from {CONFIG['model_path']}...")
    model = SSNModel(CONFIG['fdim'], CONFIG['nspix'], CONFIG['niter']).to(CONFIG['device'])
    
    if not os.path.exists(CONFIG['model_path']):
        print(f"Error: {CONFIG['model_path']} not found.")
        exit()

    try:
        state_dict = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Load weight failed ({e}). Proceeding with random weights (for debugging only).")

    model.eval()
    return model

def build_input_tensor(img_numpy, color_scale, pos_scale_init, nspix):
    img_tensor = torch.from_numpy(img_numpy.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(CONFIG['device']) 
    b, c, height, width = img_tensor.shape
    device = img_tensor.device
    
    nspix_per_axis = int(math.sqrt(nspix))
    scale_h = nspix_per_axis/height if height > 0 else 1
    scale_w = nspix_per_axis/width if width > 0 else 1
    pos_scale = pos_scale_init * max(scale_h, scale_w)

    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    coords = torch.stack([yy, xx], 0).unsqueeze(0).float()

    inputs = torch.cat([color_scale * img_tensor, pos_scale * coords], 1)
    return inputs

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input dir {INPUT_DIR} not found.")
        return

    img_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(img_files)} images. Starting inference...")

    for img_name in tqdm(img_files):
        img_path = os.path.join(INPUT_DIR, img_name)
        save_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + '.npy')
        
        if os.path.exists(save_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        orig_h, orig_w = img.shape[:2]
        
        img_resized = cv2.resize(img, INFER_SIZE, interpolation=cv2.INTER_LINEAR)
        inputs = build_input_tensor(img_resized, CONFIG['color_scale'], CONFIG['pos_scale'], CONFIG['nspix'])

        with torch.no_grad():
            _, hard_labels, _ = model(inputs)
            h_map = hard_labels.view(INFER_SIZE[1], INFER_SIZE[0]).cpu().numpy().astype(np.int32)

        ssn_map_full = cv2.resize(h_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        np.save(save_path, ssn_map_full)

    print("Done!")

if __name__ == '__main__':
    main()