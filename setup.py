#!/usr/bin/env python

# Augmented from the PyGeoN setup.py
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="tpysa",
    version="0.1",
    license="GPL",
    keywords=["TPSA"],
    author="Wietse M. Boon",
    install_requires=required,
    description="A Python implementation of TPSA",
    maintainer="Wietse M. Boon",
    maintainer_email="wibo@norceresearch.no",
    platforms=["Linux", "Windows"],
    package_data={"tpysa": ["py.typed"]},
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    zip_safe=False,
)
