from setuptools import setup, Extension

module = Extension(
    "vector_add_wrapper",
    sources=["vector_add_wrapper.c"],
)

setup(
    name="vector_add_wrapper",
    version="1.0",
    description="Python binding for CUDA vector addition",
    ext_modules=[module],
)
