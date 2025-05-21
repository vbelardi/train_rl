#!/bin/bash
#SBATCH --job-name=sim6
#SBATCH --output=log.out_sim6
#SBATCH --error=log.err_sim6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim_6.py
