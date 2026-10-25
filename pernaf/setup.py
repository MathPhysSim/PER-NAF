from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
readme_path = path.join(this_directory, "README.md")
long_description = ""
if path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="pernaf",
    version="0.2.0",
    author="Simon Hirlaender",
    author_email="simon.hirlaender@plus.ac.at",
    description=(
        "High-performance Normalized Advantage Function (NAF / NAF2 / PER-NAF / PER-NAF2) "
        "Reinforcement Learning with JAX XLA + Numba LLVM + PyTorch backends"
    ),
    url="https://github.com/MathPhysSim/PER-NAF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numba>=0.58.0",
        "jax>=0.4.20",
        "optax>=0.1.7",
        "tqdm>=4.65.0",
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
