from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='blockedell',
    ext_modules=[
        CUDAExtension(
            name='blockedell',
            sources=['bindings.cpp', 'functions.cu'],
            include_dirs=[
                # Torch headers
                f"{os.environ['CONDA_PREFIX']}/lib/python3.10/site-packages/torch/include",
                f"{os.environ['CONDA_PREFIX']}/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
                "/cm/shared/apps/linux-ubuntu22.04-zen2/cuda/11.8.0/include"
            ],
            library_dirs=[
                f"{os.environ['CONDA_PREFIX']}/lib/python3.10/site-packages/torch/lib",
                "/cm/shared/apps/linux-ubuntu22.04-zen2/cuda/11.8.0/lib64"
            ],
            libraries=['torch', 'torch_cpu', 'c10', 'cudart', 'c10_cuda', 'torch_cuda'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': [
                    '-O3', '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-D_GLIBCXX_USE_CXX11_ABI=0'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
