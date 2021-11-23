#!/bin/bash
#
#SBATCH --job-name=combined_lstm_test
#
#SBATCH -p gpu
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

module purge
ml cuda/11.2
ml cudnn/8.1

eval "$(conda shell.bash hook)"

# conda activate /home/groups/gorle/miniconda3/envs/pytorch_env
conda activate /home/groups/gorle/miniconda3/envs/tensorflow_env

# python generate_dataset_split.py
python combined_dataset_lstm_base_model.py
# python individual_dataset_lstm_base_model.py
# python individual_dataset_lstm_multifit_model.py

echo "combined_lstm_base_model done"
date
