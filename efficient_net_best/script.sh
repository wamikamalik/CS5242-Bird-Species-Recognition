#!/bin/sh

#SBATCH --job-name=efficient_net
#SBATCH --output=gpu.%j.out
#SBATCH --error=gpu.%j.err
#SBATCH --gpus=8
#SBATCH --time=900
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL,TIMEOUT
#SBATCH --mail-user=e0426346@comp.nus.edu.sg

pip install pytz
python efficient_net.py
