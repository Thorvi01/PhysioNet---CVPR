#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -J ecg                              # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 16                               # Number of tasks
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file

# Your program/command here

module load miniconda3
source activate myConda