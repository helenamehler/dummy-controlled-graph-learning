"""Build configuration for the Cython extension modules.

The package metadata lives in ``pyproject.toml``; this file only declares the
two compiled extensions. They are intentionally kept free of platform-specific
BLAS/Accelerate linking so the project builds on Linux, macOS and Windows. The
heavy linear algebra is delegated to NumPy/SciPy, and the ``prange`` loop in
``_graph_utils`` runs serially when the compiler is built without OpenMP.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "dummy_ggm._graph_utils",
        ["src/dummy_ggm/_graph_utils.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "dummy_ggm._helpers",
        ["src/dummy_ggm/_helpers.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
