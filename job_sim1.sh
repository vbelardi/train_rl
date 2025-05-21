#!/bin/bash
#SBATCH --job-name=sim1
#SBATCH --output=log.out_sim1
#SBATCH --error=log.err_sim1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim_1.py
