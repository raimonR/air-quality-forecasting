#!/bin/bash
#
#SBATCH --job-name=lstm_tuning
#
#SBATCH -p gpu
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1

module purge
ml cuda/11.2
ml cudnn/8.1

eval "$(conda shell.bash hook)"

conda activate /home/groups/gorle/miniconda3/envs/tensorflow_env

python combined_dataset_lstm_base_model.py
