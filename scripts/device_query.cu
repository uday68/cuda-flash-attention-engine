#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "\n================ Device " << i << ": " << prop.name << " ================\n";
        std::cout << "Compute Capability:            " << prop.major << "." << prop.minor << "\n";
        std::cout << "SM Count (Multiprocessors):    " << prop.multiProcessorCount << "\n";
        std::cout << "Total Global Memory:           " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "Shared Memory per Block:       " << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "Shared Memory per SM:          " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
        std::cout << "Registers per Block (32-bit):  " << prop.regsPerBlock << "\n";
        std::cout << "Registers per SM (32-bit):     " << prop.regsPerMultiprocessor << "\n";
        std::cout << "Warp Size:                     " << prop.warpSize << "\n";
        std::cout << "Max Threads per Block:         " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max Threads per SM:            " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Memory Bus Width:              " << prop.memoryBusWidth << " bits\n";
        std::cout << "Memory Clock Rate:             " << prop.memoryClockRate / 1000 << " MHz\n";
        std::cout << "L2 Cache Size:                 " << prop.l2CacheSize / 1024 << " KB\n";
    }
    return 0;
}

