import torch
import torch.nn.functional as F
import math

class PairwiseDistFunction:
    @staticmethod
    def apply(pixel_features, spixel_features, init_spixel_indices, num_spixels_w, num_spixels_h):
        """
        纯 PyTorch 实现，支持自动求导，彻底解决 C++ 编译和维度问题。
        Args:
            pixel_features: (B, C, H, W) or (B, C, N_pixels)
            spixel_features: (B, C, N_spixels)
            init_spixel_indices: (B, 1, H, W) or (B, H, W) or (B, N_pixels)
            num_spixels_w, num_spixels_h: int
        """
        # 1. 智能维度推断与标准化
        # ---------------------------------------------------------
        if pixel_features.dim() == 3: # (B, C, N) -> 需要 reshape 成 (B, C, H, W)
            b, c, n = pixel_features.shape
            
            # 尝试从 indices 推断 H, W
            if init_spixel_indices.dim() == 4: # (B, 1, H, W)
                h, w = init_spixel_indices.shape[2], init_spixel_indices.shape[3]
            elif init_spixel_indices.dim() == 3: # (B, H, W)
                h, w = init_spixel_indices.shape[1], init_spixel_indices.shape[2]
            else: # (B, N) 或 (H, W) - 最容易出错的情况
                # 如果 indices 是 (B, N) 且 N 匹配
                if init_spixel_indices.shape[1] == n:
                    # 假设是方形 patch (训练时常用)
                    h = int(math.sqrt(n))
                    w = n // h
                # 如果 indices 是 (H, W) 且 b=1
                elif init_spixel_indices.dim() == 2 and b == 1:
                    h, w = init_spixel_indices.shape
                else:
                    # 兜底方案：直接开方
                    h = int(math.sqrt(n))
                    w = n // h
            
            # 执行 Reshape
            pixel_features = pixel_features.view(b, c, h, w)
            # 同时把 indices 也还原成空间结构，方便后续计算坐标
            if init_spixel_indices.dim() == 2:
                init_spixel_indices = init_spixel_indices.view(b, h, w)
        
        # 此时 pixel_features 必定是 (B, C, H, W)
        b, c, h, w = pixel_features.shape
        device = pixel_features.device

        # 确保 indices 是 (B, H, W)
        if init_spixel_indices.dim() == 4:
            init_spixel_indices = init_spixel_indices.squeeze(1)
        
        init_spixel_indices = init_spixel_indices.long()

        # 2. 准备 9 宫格偏移
        # ---------------------------------------------------------
        shifts_y = torch.tensor([-1, -1, -1,  0, 0, 0,  1, 1, 1], device=device)
        shifts_x = torch.tensor([-1,  0,  1, -1, 0, 1, -1, 0, 1], device=device)

        # 3. 计算网格坐标
        # ---------------------------------------------------------
        # init_spixel_indices 是绝对 ID，转为 (row, col)
        sp_y = torch.div(init_spixel_indices, num_spixels_w, rounding_mode='floor') # (B, H, W)
        sp_x = init_spixel_indices % num_spixels_w                                  # (B, H, W)

        # 4. 遍历 9 个邻居
        # ---------------------------------------------------------
        dist_list = []
        
        # 预处理 spixel_features 以便 gather
        # (B, C, N_sp) -> (B, N_sp, C)
        sp_feat_perm = spixel_features.permute(0, 2, 1) 

        for k in range(9):
            # 计算邻居坐标
            neighbor_sp_y = sp_y + shifts_y[k]
            neighbor_sp_x = sp_x + shifts_x[k]

            # 越界掩码
            valid_mask = (neighbor_sp_y >= 0) & (neighbor_sp_y < num_spixels_h) & \
                         (neighbor_sp_x >= 0) & (neighbor_sp_x < num_spixels_w)

            # 坐标钳制 (防止索引越界报错)
            safe_y = neighbor_sp_y.clamp(0, num_spixels_h - 1)
            safe_x = neighbor_sp_x.clamp(0, num_spixels_w - 1)

            # 计算邻居 ID
            neighbor_id = safe_y * num_spixels_w + safe_x # (B, H, W)

            # Gather 特征
            # 扩展 neighbor_id 为 (B, H, W, C)
            neighbor_id_expanded = neighbor_id.unsqueeze(-1).expand(-1, -1, -1, c)
            
            # 取出特征: (B, N_sp, C) gather by (B, H, W, C) -> (B, H, W, C)
            # 注意：PyTorch gather 要求维度一致，这里我们需要把 N_sp 维对应到 H*W
            # 更好的方法是用 advanced indexing，但在 module 中 gather 更稳健
            # 我们先 flatten 空间维度：(B, H*W, C)
            # sp_feat_perm 是 (B, N_sp, C)
            # 这实际上是个 lookup table 操作。
            
            # 使用 gather 的正确姿势：
            # dim=1 是 lookup 维度。
            # neighbor_id_flat = neighbor_id.view(b, h*w, 1).expand(-1, -1, c)
            # neighbor_feat = torch.gather(sp_feat_perm, 1, neighbor_id_flat) # (B, H*W, C)
            # neighbor_feat = neighbor_feat.view(b, h, w, c)
            
            neighbor_id_flat = neighbor_id.view(b, h*w, 1).expand(-1, -1, c)
            neighbor_feat = torch.gather(sp_feat_perm, 1, neighbor_id_flat)
            neighbor_feat = neighbor_feat.view(b, h, w, c).permute(0, 3, 1, 2) # (B, C, H, W)

            # 计算距离
            diff = pixel_features - neighbor_feat
            dist_sq = torch.sum(diff * diff, dim=1) # (B, H, W)

            # 处理越界：设为 1e6
            dist_sq = torch.where(valid_mask, dist_sq, torch.tensor(1e6, device=device))

            dist_list.append(dist_sq)

        # 5. 输出结果
        # ---------------------------------------------------------
        # (B, 9, H, W)
        output = torch.stack(dist_list, dim=1)

        # 还原为展平格式 (B, 9, N)，以匹配 SSN 的后续操作
        output = output.view(b, 9, -1)

        return output