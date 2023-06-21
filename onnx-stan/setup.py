# setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "onnx-stan",
        ["onnx-stan/cpp/onnx-stan.cpp"],
    ),
]

setup(
    name="onnx-stan",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
