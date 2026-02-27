"""Setup script for hybrid-lens-design package."""

from setuptools import setup, find_packages

setup(
    name="hybrid-lens-design",
    version="0.1.0",
    packages=find_packages(include=["src*", "scripts*"]),
    python_requires=">=3.12",
)
