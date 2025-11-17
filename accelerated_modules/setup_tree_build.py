from setuptools import setup, Extension
import pybind11
import os

here = os.path.abspath(os.path.dirname(__file__))

ext = Extension(
    'vocab_tree_match',
    sources=[os.path.join(here, 'vocab_tree_match.cpp')],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++11', '-O3']
)

setup(
    name='vocab_tree_match',
    ext_modules=[ext]
)
