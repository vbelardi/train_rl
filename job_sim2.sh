#!/bin/bash
#SBATCH --job-name=sim2
#SBATCH --output=log.out_sim2
#SBATCH --error=log.err_sim2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim_2.py
