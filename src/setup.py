from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='blockedell',
    ext_modules=[
        CUDAExtension(
            name='blockedell',
            sources=['bindings.cpp', 'functions.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
