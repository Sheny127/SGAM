source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define CUDA_NUM_THREADS 1024

// CUDA kernel to compute pairwise squared L2 distance between pixels and neighboring superpixels
__global__ void pair_wise_distance_kernel(
    const float* pixel_features,
    const float* spixel_features,
    const int* init_spixel_indices,
    float* dist_matrix,
    int d,
    int h,
    int w,
    int n_spixels,
    int num_spixels_w,
    int num_spixels_h
) {
    // 1. Calculate global 1D thread index for the pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * w) return;

    // 2. Identify the superpixel ID and its spatial row/col indices for the current pixel
    int spixel_id = init_spixel_indices[idx];
    int spixel_r = spixel_id / num_spixels_w;
    int spixel_c = spixel_id % num_spixels_w;

    // 3. Iterate over the 3x3 spatial neighborhood (9 neighbors) of the superpixel
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
             int curr_sr = spixel_r + i;
             int curr_sc = spixel_c + j;

             // 4. Validate boundary conditions for neighbor superpixels
             if (curr_sr >= 0 && curr_sr < num_spixels_h && curr_sc >= 0 && curr_sc < num_spixels_w) {
                 int curr_spixel_id = curr_sr * num_spixels_w + curr_sc;
                 
                 // 5. Calculate squared L2 distance across feature channels (d)
                 float dist = 0;
                 for (int k = 0; k < d; k++) {
                     float diff = pixel_features[k * h * w + idx] - spixel_features[k * n_spixels + curr_spixel_id];
                     dist += diff * diff;
                 }
                 
                 // 6. Map the 3x3 neighbor offset to a flat 1D index in [0, 8]
                 int neighbor_idx = (i + 1) * 3 + (j + 1);
                 dist_matrix[neighbor_idx * h * w + idx] = dist;
             }
        }
    }
}

// C++ wrapper to configure and launch the CUDA kernel
torch::Tensor forward_cuda(
    torch::Tensor pixel_features,
    torch::Tensor spixel_features,
    torch::Tensor init_spixel_indices,
    int num_spixels_w,
    int num_spixels_h) 
{
    // Extract dimensions from input tensors
    auto d = pixel_features.size(0);
    auto h = pixel_features.size(1);
    auto w = pixel_features.size(2);
    auto n_spixels = spixel_features.size(1);
    
    auto options = torch::TensorOptions().dtype(pixel_features.dtype()).device(torch::kCUDA);
    
    // Initialize the distance matrix with 1e6 (large value) to handle out-of-boundary neighbors
    auto dist_matrix = torch::full({9, h, w}, 1e6, options);

    // Calculate grid and block dimensions for CUDA kernel launch
    int threads = CUDA_NUM_THREADS;
    int blocks = (h * w + threads - 1) / threads;

    // Retrieve the active PyTorch CUDA stream to ensure asynchronous safety
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch the CUDA kernel
    pair_wise_distance_kernel<<<blocks, threads, 0, stream>>>(
        pixel_features.data_ptr<float>(),
        spixel_features.data_ptr<float>(),
        init_spixel_indices.data_ptr<int>(),
        dist_matrix.data_ptr<float>(),
        d, h, w, n_spixels, num_spixels_w, num_spixels_h
    );

    return dist_matrix;
}

// PyBind11 interface binding C++ function to Python extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "pair_wise_distance forward");
}
"""
