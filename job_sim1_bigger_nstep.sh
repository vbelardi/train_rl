#!/bin/bash
#SBATCH --job-name=sim1_bigger_nstep
#SBATCH --output=log.out_sim1_bigger_nstep
#SBATCH --error=log.err_sim1_bigger_nstep
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim1_bigger_nstep.py
