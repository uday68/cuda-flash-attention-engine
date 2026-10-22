#include <iostream>
#include "common.cuh"

__global__ void test_kernel(float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = idx * 2.0f;
    }
}

int main() {
    std::cout << "Testing CUDA development environment...\n";

    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Host memory
    float* h_out = (float*)malloc(bytes);

    // Device memory
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Warm-up & Timing
    GpuTimer timer;
    timer.start();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    test_kernel<<<blocks, threads>>>(d_out, N);
    CUDA_CHECK_LAST_ERROR();

    timer.stop();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Validate
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != i * 2.0f) {
            passed = false;
            break;
        }
    }

    std::cout << "Kernel execution time: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Environment test: " << (passed ? "PASSED [OK]" : "FAILED") << "\n";

    CUDA_CHECK(cudaFree(d_out));
    free(h_out);

    return passed ? 0 : 1;
}

