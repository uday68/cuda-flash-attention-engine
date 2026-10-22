#pragma once

#include "common.cuh"
#include <cmath>
#include <cfloat>

namespace attention {

// ============================================================================
// Phase 4 & 5: Fused FlashAttention Forward Kernel
// 
// Algorithmic Structure:
//   1. IO-Aware Tiling: Q, K, V are tiled into on-chip SRAM (Shared Memory).
//   2. Online Softmax: Running max (m) and running sum (l) allow streaming
//      normalization without materializing the N x N attention matrix in HBM.
//   3. Coalesced Global Memory Transactions: Consecutive threads access
//      consecutive elements along the head dimension d.
//   4. O(N) Global Memory Traffic: Memory footprint drops from O(N^2) to O(N).
// ============================================================================

constexpr int BR = 32;        // Sequence tile size for Q
constexpr int BC = 32;        // Sequence tile size for K, V
constexpr int HEAD_DIM = 64;  // Standard head dimension d

__global__ void flash_attention_fwd_kernel(
    const float* __restrict__ Q,  // [B, H, N, d]
    const float* __restrict__ K,  // [B, H, N, d]
    const float* __restrict__ V,  // [B, H, N, d]
    float* __restrict__ O,        // [B, H, N, d]
    int N,
    int d,
    float scale
) {
    // Shared Memory allocation:
    // s_Q: [32][64] = 8 KB
    // s_K: [32][64] = 8 KB
    // s_V: [32][64] = 8 KB
    // Total: 24 KB SRAM
    __shared__ float s_Q[BR][HEAD_DIM];
    __shared__ float s_K[BC][HEAD_DIM];
    __shared__ float s_V[BC][HEAD_DIM];

    // Thread & Block coordinates
    // Block size is (BR, 1) -> threadIdx.x is row i in [0..BR-1]
    int tid = threadIdx.x; // row in the current Q tile [0..BR-1]
    int q_block_idx = blockIdx.x; // index of Q tile along N
    int batch_head_idx = blockIdx.y; // (b * H + h)

    int q_row_global = q_block_idx * BR + tid;

    int bh_offset = batch_head_idx * N * d;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    // Registers for Online Softmax state (private to each thread)
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_reg[HEAD_DIM];

    #pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k) {
        o_reg[k] = 0.0f;
    }

    // Step 1: Load Q block into Shared Memory
    // Thread tid loads its corresponding row of Q (all d elements)
    if (q_row_global < N) {
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) {
            s_Q[tid][k] = Q_bh[q_row_global * d + k];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) {
            s_Q[tid][k] = 0.0f;
        }
    }
    __syncthreads();

    int num_kv_blocks = (N + BC - 1) / BC;

    // Step 2: Loop over all K, V blocks in SRAM
    for (int j = 0; j < num_kv_blocks; ++j) {
        // Cooperatively load K and V tiles into Shared Memory
        // Thread tid loads row tid of K and V
        int kv_row_global = j * BC + tid;
        if (kv_row_global < N) {
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; ++k) {
                s_K[tid][k] = K_bh[kv_row_global * d + k];
                s_V[tid][k] = V_bh[kv_row_global * d + k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; ++k) {
                s_K[tid][k] = 0.0f;
                s_V[tid][k] = 0.0f;
            }
        }
        __syncthreads();

        // If this thread represents a valid Q row, compute tile attention
        if (q_row_global < N) {
            // Compute S_ij = (Q_i * K_col^T) * scale for all col in [0..BC-1]
            float s_tile[BC];
            float m_tile = -FLT_MAX;

            #pragma unroll
            for (int col = 0; col < BC; ++col) {
                int global_col = j * BC + col;
                if (global_col < N) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < HEAD_DIM; ++k) {
                        dot += s_Q[tid][k] * s_K[col][k];
                    }
                    float val = dot * scale;
                    s_tile[col] = val;
                    if (val > m_tile) {
                        m_tile = val;
                    }
                } else {
                    s_tile[col] = -FLT_MAX;
                }
            }

            // --- Online Softmax Update ---
            // 1. Compute new maximum
            float m_new = fmaxf(m_prev, m_tile);

            // 2. Exponential scaling factor for prior accumulation
            float alpha = (m_prev == -FLT_MAX) ? 0.0f : expf(m_prev - m_new);

            // 3. Compute exponentials for current tile
            float p_tile[BC];
            float sum_p = 0.0f;
            #pragma unroll
            for (int col = 0; col < BC; ++col) {
                if (j * BC + col < N) {
                    float p = expf(s_tile[col] - m_new);
                    p_tile[col] = p;
                    sum_p += p;
                } else {
                    p_tile[col] = 0.0f;
                }
            }

            // 4. Update running sum l
            float l_new = alpha * l_prev + sum_p;

            // 5. Update running output vector O = alpha * O + P_tile * V_tile
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; ++k) {
                float v_accum = 0.0f;
                #pragma unroll
                for (int col = 0; col < BC; ++col) {
                    v_accum += p_tile[col] * s_V[col][k];
                }
                o_reg[k] = alpha * o_reg[k] + v_accum;
            }

            // Update state for next K, V tile
            m_prev = m_new;
            l_prev = l_new;
        }

        __syncthreads();
    }

    // Step 3: Final Normalization by 1/l and write to Global Memory
    if (q_row_global < N) {
        float inv_l = 1.0f / (l_prev + 1e-8f);
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) {
            O_bh[q_row_global * d + k] = o_reg[k] * inv_l;
        }
    }
}

// Host launcher
inline void launch_flash_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int B,
    int H,
    int N,
    int d,
    cudaStream_t stream = 0
) {
    if (d != HEAD_DIM) {
        std::cerr << "FlashAttention requires head_dim d == " << HEAD_DIM << std::endl;
        std::exit(EXIT_FAILURE);
    }

    float scale = 1.0f / sqrtf((float)d);
    int total_heads = B * H;

    dim3 block(BR); // 32 threads per block
    dim3 grid((N + BR - 1) / BR, total_heads);

    flash_attention_fwd_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V, d_O, N, d, scale
    );
    CUDA_CHECK_LAST_ERROR();
}

} // namespace attention

