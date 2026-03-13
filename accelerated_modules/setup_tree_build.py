from setuptools import setup, Extension
import pybind11
import os

here = os.path.abspath(os.path.dirname(__file__))

vocab_tree_ext = Extension(
    'vocab_tree_match',
    sources=[os.path.join(here, 'vocab_tree_match.cpp')],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++11', '-O3']
)

image_retrieval_ext = Extension(
    'image_retrieval_match',
    sources=[os.path.join(here, 'image_retrieval_match.cpp')],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++14', '-O3']
)

setup(
    name='vocab_tree_match',
    ext_modules=[vocab_tree_ext, image_retrieval_ext]
)