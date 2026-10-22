#pragma once

#include "common.cuh"
#include <cmath>
#include <cfloat>

namespace attention {

// ============================================================================
// Senior DevTech Level: FlashAttention-2 with Warp Shuffle Intrinsics,
// 128-bit Vectorized Memory Access (float4), and Causal Masking.
// ============================================================================

constexpr int FA2_BR = 32;       // Sequence tile size for Q
constexpr int FA2_BC = 32;       // Sequence tile size for K, V
constexpr int FA2_HEAD_DIM = 64; // Head dimension d

// Warp-level reduction for maximum value using register shuffle intrinsics
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum using register shuffle intrinsics
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused FlashAttention-2 Kernel with Causal Masking support
template <bool IS_CAUSAL>
__global__ void flash_attention_v2_fwd_kernel(
    const float* __restrict__ Q,  // [B, H, N, d]
    const float* __restrict__ K,  // [B, H, N, d]
    const float* __restrict__ V,  // [B, H, N, d]
    float* __restrict__ O,        // [B, H, N, d]
    int N,
    int d,
    float scale
) {
    // Shared Memory staging buffers
    __shared__ float s_Q[FA2_BR][FA2_HEAD_DIM];
    __shared__ float s_K[FA2_BC][FA2_HEAD_DIM];
    __shared__ float s_V[FA2_BC][FA2_HEAD_DIM];

    int tid = threadIdx.x;        // Row index inside the Q tile [0..FA2_BR-1]
    int q_block_idx = blockIdx.x; // Block along sequence length
    int batch_head_idx = blockIdx.y; // (b * H + h)

    int q_row_global = q_block_idx * FA2_BR + tid;

    int bh_offset = batch_head_idx * N * d;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    // Online Softmax tracking registers
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_reg[FA2_HEAD_DIM];

    #pragma unroll
    for (int k = 0; k < FA2_HEAD_DIM; ++k) {
        o_reg[k] = 0.0f;
    }

    // 128-bit Vectorized load for Q using float4 (16 bytes per transaction)
    // 64 floats = 16 float4s
    if (q_row_global < N) {
        const float4* Q_vec = reinterpret_cast<const float4*>(Q_bh + q_row_global * d);
        float4* s_Q_vec = reinterpret_cast<float4*>(s_Q[tid]);
        #pragma unroll
        for (int k = 0; k < FA2_HEAD_DIM / 4; ++k) {
            s_Q_vec[k] = Q_vec[k];
        }
    } else {
        float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4* s_Q_vec = reinterpret_cast<float4*>(s_Q[tid]);
        #pragma unroll
        for (int k = 0; k < FA2_HEAD_DIM / 4; ++k) {
            s_Q_vec[k] = zero4;
        }
    }
    __syncthreads();

    // Determine the range of K, V blocks to process
    // In Causal mode, we can completely skip blocks strictly in the future!
    int num_kv_blocks = (N + FA2_BC - 1) / FA2_BC;
    int max_kv_block = num_kv_blocks;
    if (IS_CAUSAL) {
        // Upper bound on KV blocks: only up to the current Q tile
        max_kv_block = min(num_kv_blocks, (q_block_idx * FA2_BR + FA2_BR + FA2_BC - 1) / FA2_BC);
    }

    // Loop over valid K, V tiles
    for (int j = 0; j < max_kv_block; ++j) {
        int kv_row_global = j * FA2_BC + tid;

        // 128-bit Vectorized load for K and V
        if (kv_row_global < N) {
            const float4* K_vec = reinterpret_cast<const float4*>(K_bh + kv_row_global * d);
            const float4* V_vec = reinterpret_cast<const float4*>(V_bh + kv_row_global * d);
            float4* s_K_vec = reinterpret_cast<float4*>(s_K[tid]);
            float4* s_V_vec = reinterpret_cast<float4*>(s_V[tid]);

            #pragma unroll
            for (int k = 0; k < FA2_HEAD_DIM / 4; ++k) {
                s_K_vec[k] = K_vec[k];
                s_V_vec[k] = V_vec[k];
            }
        } else {
            float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4* s_K_vec = reinterpret_cast<float4*>(s_K[tid]);
            float4* s_V_vec = reinterpret_cast<float4*>(s_V[tid]);

            #pragma unroll
            for (int k = 0; k < FA2_HEAD_DIM / 4; ++k) {
                s_K_vec[k] = zero4;
                s_V_vec[k] = zero4;
            }
        }
        __syncthreads();

        if (q_row_global < N) {
            float s_tile[FA2_BC];
            float m_tile = -FLT_MAX;

            #pragma unroll
            for (int col = 0; col < FA2_BC; ++col) {
                int global_col = j * FA2_BC + col;

                // Causal masking check: col > row is illegal
                if (global_col < N && (!IS_CAUSAL || global_col <= q_row_global)) {
                    // Vectorized dot product Q_i . K_col
                    const float4* q_v = reinterpret_cast<const float4*>(s_Q[tid]);
                    const float4* k_v = reinterpret_cast<const float4*>(s_K[col]);

                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < FA2_HEAD_DIM / 4; ++k) {
                        float4 q4 = q_v[k];
                        float4 k4 = k_v[k];
                        dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
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
            float m_new = fmaxf(m_prev, m_tile);
            float alpha = (m_prev == -FLT_MAX) ? 0.0f : expf(m_prev - m_new);

            float p_tile[FA2_BC];
            float sum_p = 0.0f;
            #pragma unroll
            for (int col = 0; col < FA2_BC; ++col) {
                int global_col = j * FA2_BC + col;
                if (global_col < N && (!IS_CAUSAL || global_col <= q_row_global)) {
                    float p = expf(s_tile[col] - m_new);
                    p_tile[col] = p;
                    sum_p += p;
                } else {
                    p_tile[col] = 0.0f;
                }
            }

            float l_new = alpha * l_prev + sum_p;

            #pragma unroll
            for (int k = 0; k < FA2_HEAD_DIM; ++k) {
                float v_accum = 0.0f;
                #pragma unroll
                for (int col = 0; col < FA2_BC; ++col) {
                    v_accum += p_tile[col] * s_V[col][k];
                }
                o_reg[k] = alpha * o_reg[k] + v_accum;
            }

            m_prev = m_new;
            l_prev = l_new;
        }

        __syncthreads();
    }

    // Step 3: Final Normalization by 1/l and 128-bit Vectorized Store to Global Memory
    if (q_row_global < N) {
        float inv_l = 1.0f / (l_prev + 1e-8f);
        float4* O_vec = reinterpret_cast<float4*>(O_bh + q_row_global * d);

        #pragma unroll
        for (int k = 0; k < FA2_HEAD_DIM / 4; ++k) {
            float4 out4;
            out4.x = o_reg[k * 4 + 0] * inv_l;
            out4.y = o_reg[k * 4 + 1] * inv_l;
            out4.z = o_reg[k * 4 + 2] * inv_l;
            out4.w = o_reg[k * 4 + 3] * inv_l;
            O_vec[k] = out4; // Vectorized 128-bit store instruction (STG.E.128)
        }
    }
}

// Host launcher for FlashAttention-2
template <bool IS_CAUSAL = false>
inline void launch_flash_attention_v2(
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
    if (d != FA2_HEAD_DIM) {
        std::cerr << "FlashAttention-2 requires head_dim d == " << FA2_HEAD_DIM << std::endl;
        std::exit(EXIT_FAILURE);
    }

    float scale = 1.0f / sqrtf((float)d);
    int total_heads = B * H;

    dim3 block(FA2_BR);
    dim3 grid((N + FA2_BR - 1) / FA2_BR, total_heads);

    flash_attention_v2_fwd_kernel<IS_CAUSAL><<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V, d_O, N, d, scale
    );
    CUDA_CHECK_LAST_ERROR();
}

} // namespace attention
