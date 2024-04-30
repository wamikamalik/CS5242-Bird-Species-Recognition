#!/bin/bash
# Exercise 2 submission script – submit.sh
# Below, is the queue

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:59:59
#PBS -P personal-e1144115
#PBS -N efficient_net_neg_entropy
# Commands start here
module load miniforge3/23.10 
cd ${PBS_O_WORKDIR}
conda activate CS5242_Birds
export PYTHONPATH=$PYTHONPATH:pwd
python train_test/train.py --epoch 8 --beta 2.0 --model "efficient" --margin 0.4

#############
# parser = argparse.ArgumentParser(description="Efficient Net")
# parser.add_argument("--model", type=str, default="efficient", choices=["efficient", "google", "mobile"],
#                     help = "efficient") 
# parser.add_argument("--epoch", type = int, default=30,
#                     help="epochs")
# parser.add_argument("--temperature", type = int, default=1000,
#                     help="temp")
# parser.add_argument("--data", type=int,  default=4,
#                     help = "data volume")
# parser.add_argument("--mode", type=str, default="OOD",
#                     help = "ood or normal")
# parser.add_argument("--idbs", type=int, default=64,
#                     help = "id batch size")
# parser.add_argument("--oodbs", type=int, default=16,
#                     help = "ood batch size")
# parser.add_argument("--dir", type=str, default="nscc", choices=["local", "nscc", "custom"],
#                     help = "img dir")
# parser.add_argument("--beta", type=float, default=0.5,
#                     help = "beta hp") 
# parser.add_argument("--eps", type=float, default=10e-5,
#                     help = "beta hp") 

# args = parser.parse_args()