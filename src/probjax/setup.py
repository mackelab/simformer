#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

NAME = "probjax"
DESCRIPTION = "Jax library for probabilistic computations"
URL = "TODO"
EMAIL = "TODO"
AUTHOR = "Anonymous"
REQUIRES_PYTHON = ">=3.7.0"

REQUIRED = [
    "numpy==1.26.3",
    "scipy==1.11.4",
    "matplotlib",
    "jax==0.4.23",
    "jaxlib==0.4.23",
    "dm-haiku<=0.0.11",
    "optax",
    "ott-jax",
    "networkx",
    "IPython",
]



EXTRAS = {
    "cuda": [
            'jax[cuda12]==0.4.23',
        ],
    "cuda_fix": [
            'nvidia-cudnn-cu12==8.9.7.29',
            'jaxlib@https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.23+cuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl',
        ],
    "dev": [
        "autoflake",
        "black",
        "flake8",
        "isort>5.0.0",
        "ipdb",
        "pytest",
        "pytest-plt",
        "typeguard",
    ]
}


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
# about = {}
# project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
# with open(os.path.join(here, project_slug, "__version__.py")) as f:
#     exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ],
    license="MIT",
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    cmdclass={
        "upload": UploadCommand,
    },
)
