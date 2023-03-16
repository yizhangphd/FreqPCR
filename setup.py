from setuptools import setup, find_packages
from typing import List
import os
from typing import List, Optional

import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension, CppExtension

def get_existing_ccbin(nvcc_args: List[str]) -> Optional[str]:
    """
    Given a list of nvcc arguments, return the compiler if specified.

    Note from CUDA doc: Single value options and list options must have
    arguments, which must follow the name of the option itself by either
    one of more spaces or an equals character.
    """
    last_arg = None
    for arg in reversed(nvcc_args):
        if arg == "-ccbin":
            return last_arg
        if arg.startswith("-ccbin="):
            return arg[7:]
        last_arg = arg
    return None

extra_compile_args = {"cxx": ["-std=c++14"]}
define_macros = []

force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
    extension = CUDAExtension
    # sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    nvcc_args = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
    if nvcc_flags_env != "":
        nvcc_args.extend(nvcc_flags_env.split(" "))

    # It's better if pytorch can do this by default ..
    # CC = os.environ.get("CC", None)
    # if CC is not None:
    #     CC_arg = "-ccbin={}".format(CC)
    #     if CC_arg not in nvcc_args:
    #         if any(arg.startswith("-ccbin") for arg in nvcc_args):
    #             raise ValueError("Inconsistent ccbins")
    #         nvcc_args.append(CC_arg)

    
    CC = os.environ.get("CC", None)
    if CC is not None:
        existing_CC = get_existing_ccbin(nvcc_args)
        if existing_CC is None:
            CC_arg = "-ccbin={}".format(CC)
            nvcc_args.append(CC_arg)
        elif existing_CC != CC:
            msg = f"Inconsistent ccbins: {CC} and {existing_CC}"
            raise ValueError(msg)

    extra_compile_args["nvcc"] = nvcc_args
else:
    print('Cuda is not available!')


ext_modules = [
    CUDAExtension('rasterizer._C', [
        'rasterizer/csrc/ext.cpp',
        'rasterizer/csrc/rasterize_points.cu',
        'rasterizer/csrc/rasterize_points_cpu.cpp'
    ],
        include_dirs=['rasterizer/csrc'],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
]


setup(
    name='pytorch-rasterizer',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
