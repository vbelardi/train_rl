#!/bin/bash
#SBATCH --job-name=sim1_biggerlayers_mlp
#SBATCH --output=log.out_sim1_biggerlayers_mlp
#SBATCH --error=log.err_sim1_biggerlayers_mlp
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim1_biggerlayers_mlp.py
