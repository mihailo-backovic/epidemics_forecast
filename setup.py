from setuptools import setup
import sys
import os

setup(
    name="epidemics_forecaster",
    version="0.0.1",
    description="Library for SIS model analysis of epidemiological data",
    author="Mihailo Backovic",
    packages=["EpidemicForecaster",],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas>1.0.0",
        "scipy",
        "sympy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.5"
        "Programming Language :: Python :: 3.6"
        "Programming Language :: Python :: 3.7"
    ],
    python_requires=">= 3.0.*",
)

