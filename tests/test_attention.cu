#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include "common.cuh"
#include "naive_attention.cuh"
#include "flash_attention.cuh"

// Golden CPU Reference Implementation
void cpu_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int N,
    int d
) {
    float scale = 1.0f / std::sqrt((float)d);
    std::vector<float> S(N * N);
    std::vector<float> P(N * N);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int bh_offset = (b * H + h) * N * d;
            const float* q_bh = Q + bh_offset;
            const float* k_bh = K + bh_offset;
            const float* v_bh = V + bh_offset;
            float* o_bh = O + bh_offset;

            // 1. S = Q * K^T * scale
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < d; ++k) {
                        sum += q_bh[i * d + k] * k_bh[j * d + k];
                    }
                    S[i * N + j] = sum * scale;
                }
            }

            // 2. P = softmax(S) row-wise
            for (int i = 0; i < N; ++i) {
                float max_val = -1e30f;
                for (int j = 0; j < N; ++j) {
                    if (S[i * N + j] > max_val) {
                        max_val = S[i * N + j];
                    }
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    float exp_val = std::exp(S[i * N + j] - max_val);
                    P[i * N + j] = exp_val;
                    sum_exp += exp_val;
                }

                for (int j = 0; j < N; ++j) {
                    P[i * N + j] /= sum_exp;
                }
            }

            // 3. O = P * V
            for (int i = 0; i < N; ++i) {
                for (int col = 0; col < d; ++col) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        sum += P[i * N + j] * v_bh[j * d + col];
                    }
                    o_bh[i * d + col] = sum;
                }
            }
        }
    }
}

float compute_max_abs_diff(const float* a, const float* b, size_t total_elements) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_elements; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  FlashAttention Engine: Correctness & Verification Test \n";
    std::cout << "=========================================================\n\n";

    // Test parameters
    const int B = 2;          // Batch size
    const int H = 4;          // Number of heads
    const int N = 256;        // Sequence length
    const int d = 64;         // Head dimension (d = 64)

    size_t qkv_elements = (size_t)B * H * N * d;
    size_t matrix_s_elements = (size_t)B * H * N * N;
    size_t qkv_bytes = qkv_elements * sizeof(float);
    size_t s_bytes = matrix_s_elements * sizeof(float);

    std::cout << "Config: Batch=" << B << ", Heads=" << H << ", SeqLen=" << N << ", HeadDim=" << d << "\n";
    std::cout << "Allocating Host Memory...\n";

    std::vector<float> h_Q(qkv_elements);
    std::vector<float> h_K(qkv_elements);
    std::vector<float> h_V(qkv_elements);
    std::vector<float> h_O_cpu(qkv_elements);
    std::vector<float> h_O_naive(qkv_elements);
    std::vector<float> h_O_flash(qkv_elements);

    // Initialize with standard normal random values
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < qkv_elements; ++i) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::cout << "Computing CPU Golden Reference (O(N^2))...\n";
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), B, H, N, d);

    std::cout << "Allocating Device Memory...\n";
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_S = nullptr, *d_P = nullptr;
    float *d_O_naive = nullptr, *d_O_flash = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_S, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_P, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_O_naive, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_O_flash, qkv_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));

    std::cout << "Running Naive Attention Kernel...\n";
    attention::launch_naive_attention(d_Q, d_K, d_V, d_S, d_P, d_O_naive, B, H, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_naive.data(), d_O_naive, qkv_bytes, cudaMemcpyDeviceToHost));

    std::cout << "Running Fused FlashAttention Kernel...\n";
    attention::launch_flash_attention(d_Q, d_K, d_V, d_O_flash, B, H, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_flash.data(), d_O_flash, qkv_bytes, cudaMemcpyDeviceToHost));

    // Numerical Verification
    float diff_naive_cpu = compute_max_abs_diff(h_O_naive.data(), h_O_cpu.data(), qkv_elements);
    float diff_flash_cpu = compute_max_abs_diff(h_O_flash.data(), h_O_cpu.data(), qkv_elements);
    float diff_flash_naive = compute_max_abs_diff(h_O_flash.data(), h_O_naive.data(), qkv_elements);

    std::cout << "\n------------------- Verification Results -------------------\n";
    std::cout << "Max Absolute Diff (Naive vs CPU):       " << diff_naive_cpu << "\n";
    std::cout << "Max Absolute Diff (Flash vs CPU):       " << diff_flash_cpu << "\n";
    std::cout << "Max Absolute Diff (Flash vs Naive):     " << diff_flash_naive << "\n";

    const float tolerance = 1e-4f;
    bool passed = (diff_flash_cpu < tolerance) && (diff_naive_cpu < tolerance);

    std::cout << "Status: " << (passed ? "ALL TESTS PASSED [SUCCESS]!" : "TEST FAILED [ERROR]") << "\n";
    std::cout << "------------------------------------------------------------\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_O_naive));
    CUDA_CHECK(cudaFree(d_O_flash));

    return passed ? 0 : 1;
}

