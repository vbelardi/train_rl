#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=log.out_final
#SBATCH --error=log.err_final
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python final_sim.py