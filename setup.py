from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "Controller",
        ["Controller.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "ConnectedMatterAgent",
        ["ConnectedMatterAgent.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="ConnectedMatterSimulation",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
        }
    ),
)