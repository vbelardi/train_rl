#!/bin/bash
#SBATCH --job-name=sim1_small_gamma
#SBATCH --output=log.out_sim1_small_gamma
#SBATCH --error=log.err_sim1_small_gamma
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim1_small_gamma.py
