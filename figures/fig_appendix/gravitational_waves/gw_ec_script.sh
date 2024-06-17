#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-galvani
#SBATCH --output=job_output_%j.txt

python gw_coverage3.py
