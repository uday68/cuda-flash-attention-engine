# CUDA FlashAttention Engine

A modular CUDA implementation of the **FlashAttention forward pass** for NVIDIA GPUs.

This repository implements memory-efficient attention from first principles using tiled matrix multiplication, shared-memory staging, online softmax, optional causal masking, and fused CUDA kernels. It also includes a CPU reference implementation, correctness tests, CUDA benchmarks, device inspection utilities, and an experimental PyTorch C++/ATen binding.

> **Project status:** Experimental research and educational implementation. This project is intended for CUDA kernel development, GPU performance analysis, and systems-level study. It is not a drop-in replacement for optimized production libraries such as NVIDIA cuDNN or the latest FlashAttention implementations.

## Scope

| Capability | Status |
|---|---|
| Attention forward pass | Supported |
| Attention backward pass | Not implemented |
| Float32 execution | Supported by the included tests and benchmarks |
| Online softmax | Supported |
| Shared-memory tiling | Supported |
| Causal masking | Supported by the FA-2 benchmark path |
| 128-bit vectorized loads | Supported by the FA-2 benchmark path |
| NVIDIA Ampere `sm_86` | Primary target |
| PyTorch C++/ATen extension | Experimental |
| Automated multi-GPU CI | Not currently provided |

The current kernels are tuned for the configurations used by the included tests and benchmarks. Head dimensions, tile sizes, tensor layouts, and supported shapes may be constrained by the kernel configuration.

## Algorithm

Standard scaled dot-product attention is commonly expressed as:

\[
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

A naive implementation materializes the complete \(N \times N\) score and probability matrices. FlashAttention avoids this materialization by processing the sequence in tiles and maintaining the softmax statistics incrementally.

For each query tile and key/value tile, the kernel computes a score tile:

\[
S = QK^T
\]

It then updates the running row maximum \(m\), normalization factor \(l\), and output accumulator \(O\):

\[
m_{new} = \max(m_{old}, \max(S_{tile}))
\]

\[
\alpha = \exp(m_{old} - m_{new})
\]

\[
l_{new} = \alpha l_{old} + \sum_j \exp(S_j - m_{new})
\]

\[
O_{new} = \alpha O_{old} + \sum_j \exp(S_j - m_{new})V_j
\]

The final output is normalized using the final value of \(l\). This streaming formulation reduces global-memory traffic and avoids allocating the full attention matrix.

## Execution Model

```text
Global memory
    Q, K, V tensors
        |
        | tiled loads
        v
Shared memory
    Q tile, K tile, V tile
        |
        | tiled score and value computation
        v
Registers
    Running softmax statistics
    Output accumulators
        |
        | normalized output write
        v
Global memory
    O tensor
```

The implementation uses a tiled query/key/value execution model. Input tiles are staged in shared memory, while intermediate softmax statistics and output accumulators are maintained in registers where possible.

### High-level tile flow

```text
                    GLOBAL MEMORY
              Q, K, V: [B, H, N, d]
                         |
                    Tile loads
                         v
+---------------------------------------------------------------------+
|                     STREAMING MULTIPROCESSOR                       |
|                                                                     |
|  Shared memory:       s_Q tile   s_K tile   s_V tile                |
|                                                                     |
|  Fused computation:   S = QK^T                                     |
|                       online row-wise softmax                       |
|                       O = P V                                       |
|                                                                     |
|  Register state:      running maximum, normalization, output         |
+---------------------------------------------------------------------+
                         |
                    Final output write
                         v
                    GLOBAL MEMORY
```

## Repository Layout

```text
include/
├── common.cuh                 CUDA error checks and GPU timing utilities
├── naive_attention.cuh        Baseline attention implementation
└── flash_attention.cuh        Tiled FlashAttention forward kernel

src/
└── torch_binding.cpp          Experimental PyTorch C++/ATen bindings

tests/
├── test_env.cu                CUDA toolchain and hardware validation
└── test_attention.cu          CUDA-versus-CPU correctness validation

benchmarks/
├── benchmark_attention.cu     Baseline and FA-1 performance benchmark
└── benchmark_v2.cu            Vectorized and causal benchmark path

docs/
└── INTERVIEW_DEFENSE.md       Mathematical derivations and design notes

scripts/
└── device_query.cu            GPU capability and device inspection utility

build.bat                      Windows build helper targeting `sm_86`
CMakeLists.txt                 CMake configuration
setup.py                       PyTorch extension build configuration
```

## Requirements

The included `build.bat` workflow is intended for a Windows CUDA development environment.

Recommended prerequisites:

- NVIDIA GPU with CUDA support; the primary target is Ampere `sm_86`
- NVIDIA driver and CUDA Toolkit with `nvcc`
- A supported Microsoft C++ build toolchain
- CMake, if using the CMake workflow
- PyTorch with CUDA support, if building the experimental extension

Before compiling, verify that the CUDA compiler and GPU are visible:

```cmd
nvcc --version
nvidia-smi
```

## Building

The repository includes a simple Windows build helper. Its general form is:

```cmd
build.bat <source-file> <output-executable>
```

For example:

```cmd
build.bat tests\test_env.cu bin\test_env.exe
.\bin\test_env.exe
```

The build helper targets the architecture configured by the repository, currently `sm_86`. For a different GPU architecture, review `build.bat` and `CMakeLists.txt` before compiling.

## Correctness Validation

The correctness test compares the included CUDA implementations against a float32 CPU reference implementation.

Run the test with:

```cmd
build.bat tests\test_attention.cu bin\test_attention.exe
.\bin\test_attention.exe
```

Example configuration:

```text
Batch size:       2
Heads:            4
Sequence length:  256
Head dimension:   64
```

Example output from the included test:

```text
------------------- Verification Results -------------------
Max Absolute Diff (Naive vs CPU):       3.57628e-07
Max Absolute Diff (Flash vs CPU):       4.91738e-07
Max Absolute Diff (Flash vs Naive):     3.8743e-07
Status: ALL TESTS PASSED [SUCCESS]!
------------------------------------------------------------
```

The exact values can vary with compiler versions, GPU architecture, instruction selection, and floating-point accumulation order. Results should be evaluated using numerical tolerances rather than bitwise equality.

For broader validation, test additional combinations of:

- Batch size
- Head count
- Sequence length
- Head dimension
- Causal and non-causal execution
- Sequence lengths that are not multiples of the tile size

## Performance Benchmarks

### Baseline and FlashAttention benchmark

```cmd
build.bat benchmarks\benchmark_attention.cu bin\benchmark_attention.exe
.\bin\benchmark_attention.exe
```

### Vectorized and causal benchmark

```cmd
build.bat benchmarks\benchmark_v2.cu bin\benchmark_v2.exe
.\bin\benchmark_v2.exe
```

### Reference results

The following results were measured on an NVIDIA RTX 3050 Laptop GPU using the benchmark programs in this repository:

```text
GPU architecture: Ampere
Compute capability: sm_86
Data type: float32

SeqLen   Naive (ms)   FA-1 (ms)   FA-2 Vectorized (ms)   FA-2 Causal (ms)
----------------------------------------------------------------------------
512      3.837        2.128       1.611                 1.552
1024     15.324       8.535       6.368                 4.515
2048     61.254       34.795      25.433                15.256
```

These are reference measurements for the included implementation, not universal performance guarantees. Results may change with:

- GPU model and memory configuration
- GPU clocks, power limits, and thermal state
- CUDA Toolkit, driver, and compiler versions
- Operating system and compiler flags
- Batch size, number of heads, head dimension, and sequence length
- Warm-up and timing configuration

The benchmark compares the included kernels against the included naive CUDA baseline. It is not a direct apples-to-apples comparison with optimized cuDNN or the latest FlashAttention-2/3 kernels.

### Interpreting the optimizations

- **Tiled execution** reduces repeated global-memory traffic by reusing data in shared memory.
- **Online softmax** avoids materializing the complete attention matrix.
- **128-bit vectorized loads** use `float4`-style transfers where alignment and shape permit.
- **Causal block skipping** avoids strictly future key/value tiles in causal attention.

For example, the included benchmark measured a 4.02x speedup for the causal configuration at sequence length 2048 under its local test conditions. This value should be treated as a benchmark observation, not a general performance claim.

## Memory-Traffic Model

The naive implementation materializes an \(N \times N\) intermediate attention matrix. The FlashAttention-style implementation keeps intermediate score and probability tiles on chip and writes only the final output.

This reduces intermediate global-memory traffic, particularly as sequence length grows. The precise traffic reduction depends on tensor shape, tile dimensions, data type, masking mode, and implementation details. The benchmark source should be treated as the source of truth for the measurements reported by this repository.

## Documentation

See [docs/INTERVIEW_DEFENSE.md](docs/INTERVIEW_DEFENSE.md) for:

- Online softmax derivation
- Arithmetic intensity and memory-bandwidth analysis
- Shared-memory tile-size considerations
- CUDA execution and NVIDIA GPU architecture notes
- Design questions for reviewing and defending the implementation

## Limitations and Future Work

Current limitations include:

- Forward pass only; backward propagation is not implemented.
- The primary build configuration targets `sm_86`.
- Kernel tile sizes and supported head dimensions are configuration-dependent.
- The included correctness and performance coverage is limited.
- The PyTorch extension interface is experimental.
- No automated CI matrix currently validates multiple CUDA versions and GPU architectures.
- The benchmark does not yet provide a comprehensive comparison against vendor libraries.
- Mixed-precision execution, Tensor Core paths, and advanced asynchronous loading are not the primary focus of the current implementation.

Potential future work includes:

- Additional compute capabilities and GPU generations
- FP16/BF16 and Tensor Core implementations
- Backward-pass support
- More flexible tensor layouts and head dimensions
- CUDA Graph and asynchronous pipeline support
- Automated correctness and performance testing
- Direct integration with PyTorch's dispatcher and standard attention APIs

## License

No license file is currently included in this repository. Add an appropriate `LICENSE` file before redistributing the code or accepting external contributions.
