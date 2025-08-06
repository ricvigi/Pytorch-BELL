import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "blockedell_cu"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            '-O0',
            '-std=c++17',
            '-Wno-deprecated-gpu-targets'
        ],
    }
    # if debug_mode:
    #     extra_compile_args["cxx"].append("-g")
    #     extra_compile_args["nvcc"].append("-g")
    #     extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "cuda")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    # extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    # cuda_sources = list(glob.glob(os.path.join(this_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="BlockedELL sparse matrix CUDA extension for PyTorch",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/ricvigi/Pytorch-BELL",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)



# setup(
#     name='blockedell',
#     version='0.0.1'
#     ext_modules=[
#         CUDAExtension(
#             name='blockedell',
#             sources=['bindings.cpp', 'functions.cu'],
#             include_dirs=[
#                 # Torch headers
#                 f"{os.environ['CONDA_PREFIX']}/lib/python3.10/site-packages/torch/include",
#                 f"{os.environ['CONDA_PREFIX']}/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
#                 "/cm/shared/apps/linux-ubuntu22.04-zen2/cuda/11.8.0/include"
#             ],
#             library_dirs=[
#                 f"{os.environ['CONDA_PREFIX']}/lib/python3.10/site-packages/torch/lib",
#                 "/cm/shared/apps/linux-ubuntu22.04-zen2/cuda/11.8.0/lib64"
#             ],
#             libraries=['torch', 'torch_cpu', 'c10', 'cudart', 'c10_cuda', 'torch_cuda'],
#             extra_compile_args={
#                 'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
#                 'nvcc': [
#                     '-O3', '--expt-relaxed-constexpr',
#                     '-gencode=arch=compute_89,code=sm_89',
#                     '-D_GLIBCXX_USE_CXX11_ABI=0'
#                 ]
#             }
#         )
#     ],
#     cmdclass={'build_ext': BuildExtension}
# )
