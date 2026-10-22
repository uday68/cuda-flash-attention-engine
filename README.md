# Production FlashAttention CUDA Engine & Systems Mastery

A high-performance, modular CUDA implementation of **FlashAttention (Forward Pass)** built from first principles for NVIDIA Ampere (`sm_86`) architectures, complete with golden CPU verification, benchmarking suites, and an NVIDIA technical interview defense manual.

---

## Architecture Overview

```
                      GLOBAL MEMORY (DRAM / HBM) - 400-800 cycles
                      +-----------------------------------------+
                      |         Q, K, V Tensors [B, H, N, d]    |
                      +-----------------------------------------+
                                   |                 ^
                        Tile Loads |                 | O_final write
                             (SRAM)|                 | (SRAM)
                                   v                 |
+-----------------------------------------------------------------------------------+
|               STREAMING MULTIPROCESSOR (SM) ON-CHIP SRAM (20-30 cycles)           |
|                                                                                   |
|  [ s_Q: 32x64 (8KB) ]       [ s_K: 32x64 (8KB) ]       [ s_V: 32x64 (8KB) ]       |
|                                                                                   |
|  ------------------------- STREAMING ONLINE SOFTMAX ----------------------------  |
|  Running Max:  m_new = max(m_prev, max(S_tile))                                   |
|  Rescaling:    alpha = exp(m_prev - m_new)                                        |
|  Running Sum:  l_new = alpha * l_prev + sum(exp(S_tile - m_new))                  |
|  Accumulator:  O_new = alpha * O_prev + P_tile * s_V                              |
+-----------------------------------------------------------------------------------+
```

---

## Project Structure

```
d:\cuda_flash_attention_engine\
├── include\
│   ├── common.cuh               # Production error checks (CUDA_CHECK) & high-precision GpuTimer
│   ├── naive_attention.cuh      # Baseline 3-kernel Attention (materializes N x N matrix)
│   └── flash_attention.cuh      # Fused FlashAttention kernel with Online Softmax in SRAM
├── src\
│   └── torch_binding.cpp        # PyTorch C++ / ATen extension bindings
├── tests\
│   ├── test_env.cu              # Toolchain and hardware validation test
│   └── test_attention.cu        # Correctness test vs CPU Golden Reference (O(N^2))
├── benchmarks\
│   └── benchmark_attention.cu   # Latency, TFLOPS, and HBM memory traffic benchmark
├── docs\
│   └── INTERVIEW_DEFENSE.md     # Mathematical derivation and NVIDIA whiteboard Q&A
├── scripts\
│   └── device_query.cu          # Real-time Ampere SM and memory hardware probe
├── build.bat                    # Single-command build script targeting sm_86
├── CMakeLists.txt               # CMake configuration
└── setup.py                     # PyTorch Extension build configuration
```

---

## 1. Running the Verification Suite

Validates the numerical precision of Naive CUDA Attention and FlashAttention against a float32 CPU golden reference:

```cmd
build.bat tests\test_attention.cu bin\test_attention.exe
.\bin\test_attention.exe
```

**Results:**
```text
Config: Batch=2, Heads=4, SeqLen=256, HeadDim=64
Computing CPU Golden Reference (O(N^2))...
Running Naive Attention Kernel...
Running Fused FlashAttention Kernel...

------------------- Verification Results -------------------
Max Absolute Diff (Naive vs CPU):       3.57628e-07
Max Absolute Diff (Flash vs CPU):       4.91738e-07
Max Absolute Diff (Flash vs Naive):     3.8743e-07
Status: ALL TESTS PASSED [SUCCESS]!
------------------------------------------------------------
```

---

## 2. Running the Performance Benchmarks

### Benchmark 1: FlashAttention-1 vs Naive Attention
```cmd
build.bat benchmarks\benchmark_attention.cu bin\benchmark_attention.exe
.\bin\benchmark_attention.exe
```

### Benchmark 2: Senior DevTech FlashAttention-2 (128-bit Vectorization + Causal Masking)
```cmd
build.bat benchmarks\benchmark_v2.cu bin\benchmark_v2.exe
.\bin\benchmark_v2.exe
```

**Live Benchmark Results on NVIDIA RTX 3050 Laptop GPU (Ampere sm_86):**
```text
SeqLen  Naive(ms)       FA-1(ms)        FA-2 Vec(ms)    FA-2 Causal(ms)   Causal Speedup    
----------------------------------------------------------------------------------------------
512     3.837 ms        2.128 ms        1.611 ms        1.552 ms          2.47x
1024    15.324 ms       8.535 ms        6.368 ms        4.515 ms          3.39x
2048    61.254 ms       34.795 ms       25.433 ms       15.256 ms         4.02x
----------------------------------------------------------------------------------------------
```
* **Vectorized 128-bit Loads (`float4`)**: Cuts instruction dispatch pressure by $4\times$, boosting throughput by another 27% over FA-1.
* **Causal Block-Skipping**: Skips strictly upper-triangular KV tiles entirely, delivering **4.02x overall speedup** over naive attention at $N=2048$.

* At $N = 2048$, FlashAttention cuts HBM DRAM traffic from **266 MB down to 8 MB** (**33x reduction**) while delivering a **1.76x speedup** on mobile Ampere silicon.

---

