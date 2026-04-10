import math
import torch
from ..utils.sparse_utils import naive_sparse_bmm 

def pairwise_dist(pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height):
    device = pixel_features.device
    b, c, h, w = pixel_features.shape
    num_spixels = spixel_features.shape[-1]
    pixel_features = pixel_features.reshape(b, c, -1).permute(0, 2, 1) 
    spixel_features = spixel_features.permute(0, 2, 1)
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

    spixel_features, init_label_map = calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features_4d = pixel_features
    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1)

    for _ in range(n_iter):
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features_4d, 
            spixel_features, 
            init_label_map.view(batch_size, height, width), 
            num_spixels_width, num_spixels_height)

        affinity_matrix = (-dist_matrix).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)
        mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
        sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])
        
        try:
             spixel_features = naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features) \
                / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)
        except:

             pass
             
        spixel_features = spixel_features.permute(0, 2, 1)

    hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)
    return sparse_abs_affinity, hard_labels, spixel_features

def ssn_iter(pixel_features, num_spixels, n_iter):
    return sparse_ssn_iter(pixel_features, num_spixels, n_iter)