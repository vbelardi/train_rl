#!/bin/bash
#SBATCH --job-name=advanced
#SBATCH --output=log.out_advanced
#SBATCH --error=log.err_advanced
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python advanced_exploration_gym_fixed.py