from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "navisim.math_operations",  # Module will be navisim.math_operations
        ["navisim/_native/math_operations.cpp"],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(os.path.dirname(__file__), "navisim", "_native"),
        ],
        cxx_std=14,
    ),
]

setup(
    name="navisim",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)