#!/bin/bash
#
#SBATCH --job-name=lstm_tuning
#
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#

module purge
ml cuda/11.2
ml cudnn/8.1

eval "$(conda shell.bash hook)"

conda activate /home/groups/gorle/miniconda3/envs/tensorflow_env

python #script goes here
