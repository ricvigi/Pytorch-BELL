from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
    'nvcc': ['-O3', '--expt-relaxed-constexpr', '-Wno-deprecated-gpu-targets', '-Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include', '-ltorch -ltorch_cpu -lc10', '-Xcompiler -fopenmp -I/opt/cuda/include -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Xlinker -rpath -Xlinker libtorch/lib -ltorch -ltorch_cpu -lc10 -lcusparse']
}

setup(
    name='blockedell',
    ext_modules=[
        CUDAExtension(
            name='blockedell',
            sources=['bindings.cpp', 'functions.cu'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
