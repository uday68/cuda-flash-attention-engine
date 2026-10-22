#include <torch/extension.h>
#include <vector>
#include "flash_attention.cuh"

// PyTorch ATen forward binding for FlashAttention
torch::Tensor flash_attention_forward(
    torch::Tensor Q, // [B, H, N, d]
    torch::Tensor K, // [B, H, N, d]
    torch::Tensor V  // [B, H, N, d]
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    TORCH_CHECK(d == attention::HEAD_DIM, "Head dimension d must match attention::HEAD_DIM (64)");

    auto O = torch::empty_like(Q);

    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    attention::launch_flash_attention(d_Q, d_K, d_V, d_O, B, H, N, d);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "FlashAttention forward pass (CUDA)");
}

