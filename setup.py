#!/usr/bin/env python

import os.path
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    name="tpysa",
    version="0.0.0",
    license="GPL",
    keywords=["TPSA"],
    author="Wietse M. Boon",
    description="A Python implementation of TPSA",
    maintainer="Wietse M. Boon",
    maintainer_email="wibo@norceresearch.no",
    platforms=["Linux", "Windows"],
    package_data={"tpysa": ["py.typed"]},
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    zip_safe=False,
)
