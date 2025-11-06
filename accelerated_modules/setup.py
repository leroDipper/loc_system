from setuptools import setup, Extension
import pybind11

ext = Extension(
    'vocab_match',
    ['vocab_match.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++11', '-O3']
)

setup(
    name='vocab_match',
    ext_modules=[ext]
)