#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --output=./slurm-outputs/%x-%j.out
#SBATCH --time=4-23:00:00     # DD-HH:MM:SS
#SBATCH --mail-user=marko.huang@mail.utoronto.ca
#SBATCH --mail-type=END

module load python/3.7 cuda cudnn
module load gentoo/2020
module load gcc/9.3.0
module load rdkit/2020.09.1b1

source $HOME/env/bin/activate

jupyter nbconvert --to script train_gen.ipynb
python -u train_gen.py
