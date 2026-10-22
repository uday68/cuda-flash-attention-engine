#pragma once

#include "common.cuh"
#include <cmath>
#include <cfloat>

namespace attention {

// ============================================================================
// Phase 1: Naive CUDA Attention
// Computes:
//   1. S = Q * K^T / sqrt(d)    [N x N intermediate stored in Global Memory]
//   2. P = softmax(S)           [N x N intermediate stored in Global Memory]
//   3. O = P * V                [N x d final output]
// ============================================================================

// Kernel 1: QK^T Batched Matmul with 1/sqrt(d) scaling
__global__ void qk_gemm_naive_kernel(
    const float* __restrict__ Q,  // [B, H, N, d]
    const float* __restrict__ K,  // [B, H, N, d]
    float* __restrict__ S,        // [B, H, N, N]
    int N,
    int d,
    float scale
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // index in N (row of Q)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // index in N (row of K)

    int batch_head_idx = blockIdx.z; // (b * H + h)

    if (row < N && col < N) {
        const float* q_row = Q + (batch_head_idx * N + row) * d;
        const float* k_col = K + (batch_head_idx * N + col) * d;

        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < d; ++k) {
            sum += q_row[k] * k_col[k];
        }

        S[(batch_head_idx * N + row) * N + col] = sum * scale;
    }
}

// Kernel 2: Row-wise Safe Softmax over S (N x N)
__global__ void softmax_naive_kernel(
    const float* __restrict__ S, // [B, H, N, N]
    float* __restrict__ P,       // [B, H, N, N]
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row in N
    int batch_head_idx = blockIdx.z;

    if (row < N) {
        const float* s_row = S + (batch_head_idx * N + row) * N;
        float* p_row = P + (batch_head_idx * N + row) * N;

        // Step 1: Find row maximum for numerical stability
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            if (s_row[j] > max_val) {
                max_val = s_row[j];
            }
        }

        // Step 2: Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum_exp += expf(s_row[j] - max_val);
        }

        // Step 3: Normalize
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < N; ++j) {
            p_row[j] = expf(s_row[j] - max_val) * inv_sum;
        }
    }
}

// Kernel 3: P * V Matmul
__global__ void pv_gemm_naive_kernel(
    const float* __restrict__ P,  // [B, H, N, N]
    const float* __restrict__ V,  // [B, H, N, d]
    float* __restrict__ O,        // [B, H, N, d]
    int N,
    int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // index in N (row of P)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // index in d (col of V)
    int batch_head_idx = blockIdx.z;

    if (row < N && col < d) {
        const float* p_row = P + (batch_head_idx * N + row) * N;

        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += p_row[k] * V[(batch_head_idx * N + k) * d + col];
        }

        O[(batch_head_idx * N + row) * d + col] = sum;
    }
}

// Host dispatcher for Naive Attention
inline void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_S,
    float* d_P,
    float* d_O,
    int B,
    int H,
    int N,
    int d,
    cudaStream_t stream = 0
) {
    float scale = 1.0f / sqrtf((float)d);
    int total_heads = B * H;

    // 1. Launch QK^T: Grid (ceil(N/16), ceil(N/16), B*H), Block (16, 16)
    dim3 block_qk(16, 16);
    dim3 grid_qk((N + 15) / 16, (N + 15) / 16, total_heads);
    qk_gemm_naive_kernel<<<grid_qk, block_qk, 0, stream>>>(d_Q, d_K, d_S, N, d, scale);
    CUDA_CHECK_LAST_ERROR();

    // 2. Launch Softmax: Grid (1, ceil(N/128), B*H), Block (1, 128)
    dim3 block_sm(1, 128);
    dim3 grid_sm(1, (N + 127) / 128, total_heads);
    softmax_naive_kernel<<<grid_sm, block_sm, 0, stream>>>(d_S, d_P, N);
    CUDA_CHECK_LAST_ERROR();

    // 3. Launch PV: Grid (ceil(d/16), ceil(N/16), B*H), Block (16, 16)
    dim3 block_pv(16, 16);
    dim3 grid_pv((d + 15) / 16, (N + 15) / 16, total_heads);
    pv_gemm_naive_kernel<<<grid_pv, block_pv, 0, stream>>>(d_P, d_V, d_O, N, d);
    CUDA_CHECK_LAST_ERROR();
}

} // namespace attention

