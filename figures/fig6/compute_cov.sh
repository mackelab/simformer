#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --output=job_output_%j.txt

python compute_cov3.py
