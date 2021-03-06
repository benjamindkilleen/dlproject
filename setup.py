#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="dlproject",
    version="0.1.1",
    description="The opinionated deep learning template.",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/benjamindkilleen/dlproject",
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "hydra-core",
        "omegaconf",
        "rich",
        "numpy",
    ],
    packages=find_packages(),
)
