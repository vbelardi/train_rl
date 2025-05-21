#!/bin/bash
#SBATCH --job-name=sim1_cnnmaxpooling
#SBATCH --output=log.out_sim1_cnnmaxpooling
#SBATCH --error=log.err_sim1_cnnmaxpooling
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
python sim1_cnnmaxpooling.py
