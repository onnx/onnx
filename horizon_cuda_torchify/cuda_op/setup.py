from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_add_ext',
    ext_modules=[
        CUDAExtension(
            name='custom_add_ext',
            sources=['custom_add.cpp', 'custom_add_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
