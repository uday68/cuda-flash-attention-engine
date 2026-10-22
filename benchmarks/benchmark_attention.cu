#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "common.cuh"
#include "naive_attention.cuh"
#include "flash_attention.cuh"

struct BenchmarkResult {
    int N;
    float naive_ms;
    float flash_ms;
    float speedup;
    float flash_tflops;
    size_t hbm_naive_mb;
    size_t hbm_flash_mb;
};

BenchmarkResult run_benchmark_for_seqlen(int B, int H, int N, int d, int iters = 20) {
    size_t qkv_elements = (size_t)B * H * N * d;
    size_t matrix_s_elements = (size_t)B * H * N * N;
    size_t qkv_bytes = qkv_elements * sizeof(float);
    size_t s_bytes = matrix_s_elements * sizeof(float);

    // Theoretical FLOPs:
    // QK^T: 2 * B * H * N * N * d
    // PV:   2 * B * H * N * N * d
    // Total Attention FLOPs ≈ 4 * B * H * N^2 * d
    double total_flops = 4.0 * (double)B * (double)H * (double)N * (double)N * (double)d;

    // HBM Memory Traffic (Bytes):
    // Naive: Reads Q, K, V (3*qkv), Writes S (s_bytes), Reads S, Writes P (2*s_bytes),
    //        Reads P, Reads V, Writes O (s_bytes + 2*qkv) -> 5*qkv + 4*s_bytes
    size_t naive_traffic_bytes = 5 * qkv_bytes + 4 * s_bytes;
    // FlashAttention: Reads Q, K, V once + Writes O once -> 4 * qkv_bytes
    size_t flash_traffic_bytes = 4 * qkv_bytes;

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

    CUDA_CHECK(cudaMemset(d_Q, 0x3f, qkv_bytes));
    CUDA_CHECK(cudaMemset(d_K, 0x3f, qkv_bytes));
    CUDA_CHECK(cudaMemset(d_V, 0x3f, qkv_bytes));

    GpuTimer timer;

    // Warm-up Naive
    for (int i = 0; i < 3; ++i) {
        attention::launch_naive_attention(d_Q, d_K, d_V, d_S, d_P, d_O_naive, B, H, N, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time Naive
    timer.start();
    for (int i = 0; i < iters; ++i) {
        attention::launch_naive_attention(d_Q, d_K, d_V, d_S, d_P, d_O_naive, B, H, N, d);
    }
    timer.stop();
    float naive_ms = timer.elapsed_ms() / iters;

    // Warm-up Flash
    for (int i = 0; i < 3; ++i) {
        attention::launch_flash_attention(d_Q, d_K, d_V, d_O_flash, B, H, N, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time Flash
    timer.start();
    for (int i = 0; i < iters; ++i) {
        attention::launch_flash_attention(d_Q, d_K, d_V, d_O_flash, B, H, N, d);
    }
    timer.stop();
    float flash_ms = timer.elapsed_ms() / iters;

    float speedup = naive_ms / flash_ms;
    float flash_tflops = (float)((total_flops / (flash_ms * 1e-3)) / 1e12);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_O_naive));
    CUDA_CHECK(cudaFree(d_O_flash));

    return {
        N,
        naive_ms,
        flash_ms,
        speedup,
        flash_tflops,
        naive_traffic_bytes / (1024 * 1024),
        flash_traffic_bytes / (1024 * 1024)
    };
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "        NVIDIA Systems Engine: FlashAttention vs. Naive Attention Benchmark            \n";
    std::cout << "        Target GPU: Ampere sm_86 (RTX 3050 Laptop)                                      \n";
    std::cout << "========================================================================================\n\n";

    const int B = 1;
    const int H = 4;
    const int d = 64;

    std::vector<int> seq_lengths = {256, 512, 1024, 2048};

    std::cout << std::left 
              << std::setw(8)  << "SeqLen"
              << std::setw(16) << "Naive Time(ms)"
              << std::setw(16) << "Flash Time(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Flash TFLOPS"
              << std::setw(16) << "Naive HBM(MB)"
              << std::setw(16) << "Flash HBM(MB)"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (int N : seq_lengths) {
        BenchmarkResult r = run_benchmark_for_seqlen(B, H, N, d, 20);
        std::cout << std::left 
                  << std::setw(8)  << r.N
                  << std::setw(16) << std::fixed << std::setprecision(3) << r.naive_ms
                  << std::setw(16) << std::fixed << std::setprecision(3) << r.flash_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.flash_tflops
                  << std::setw(16) << r.hbm_naive_mb
                  << std::setw(16) << r.hbm_flash_mb
                  << "\n";
    }

    std::cout << std::string(100, '-') << "\n";
    std::cout << "Key Observation: As SeqLen scales, Naive Attention HBM traffic explodes as O(N^2),\n";
    std::cout << "while FlashAttention remains bounded at O(N) with dramatic speedups!\n";

    return 0;
}

