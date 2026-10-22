from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

include_dirs = [
    os.path.abspath("include")
]

setup(
    name="flash_attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_attention_cuda",
            sources=[
                "src/torch_binding.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["/O2", "/std:c++17"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_86",
                    "--use_fast_math",
                    "-std=c++17"
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

