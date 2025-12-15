import os
import numpy as np
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

__version__ = "0.0.1"
NVCCFLAGS = ["-O3", "-Xcompiler", "-fPIC",]

# try to get path to cuda from CUDA_HOME, otherwise use linux /usr/local/cuda
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

class BuildWithNVCC(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu") # compiler handles .cu
        super_compile = self.compiler._compile     # original _compile method

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                nvcc = os.environ.get("NVCC", "nvcc") # cuda nvcc
                cmd = [nvcc, "-c", src, "-o", obj] + cc_args + (extra_postargs or []) # build command
                self.spawn(cmd)

            else:
                super_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        # substitute new compilator with nvcc for .cu files
        self.compiler._compile = _compile
        super().build_extensions()


ext_modules = [
    Extension(
        name="fluidgpu._wrapper",
        sources=["fluidgpu/_wrapper.cu"],
        include_dirs=[
            np.get_include(),                    # numpy headers
            os.path.join(CUDA_HOME, "include"),  # cuda headers
        ],
        extra_compile_args=NVCCFLAGS,
        library_dirs=[
            os.path.join(CUDA_HOME, "lib64")
        ],
        libraries=[
            "cudart"
        ]      
    )
]

setup(
    name="fluidgpu",
    version=__version__,
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildWithNVCC},
)