from setuptools import setup, Extension
import numpy

module = Extension(
    "vector_add_np_wrapper",
    include_dirs = [numpy.get_include()],
    sources=["vector_add_np_wrapper.c"],
)

setup(
    name="vector_add_np_wrapper",
    version="1.0",
    description="Python binding for CUDA vector addition",
    ext_modules=[module],
    include_dirs=[numpy.get_include()]
)
