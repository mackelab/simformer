from setuptools import Command, find_packages, setup
import os

NAME = "scoresbibm"
VERSION = "0.0.1"
DESCRIPTION = "Score-based inference benchmark"
AUTHOR = "Anonymous"

entry_points={
    'console_scripts': [
        'scoresbi = scoresbibm.scripts:main',
    ],
}



REQUIRED = [
    "numpy",
    "matplotlib",
    "jax",
    "torch@http://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.1.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl#sha256=88f1ee550c6291af8d0417871fb7af84b86527d18bc02ac4249f07dcd84dda56",
    "torchaudio",
    "torchvision",
    "hydra-core",
    "hydra-submitit-launcher",
    "hydra-optuna-sweeper",
    "omegaconf",
    "sbi==0.22.0",
    "optuna",
    "tueplots",
    "seaborn",
    "pandas",
]

setup(name=NAME,version=VERSION, description=DESCRIPTION, author=AUTHOR,package_dir={"scoresbibm": "scoresbibm"}, install_requires=REQUIRED, entry_points=entry_points)

# Do not print the output of the following command
os.system("pip install sbibm --no-deps -q")