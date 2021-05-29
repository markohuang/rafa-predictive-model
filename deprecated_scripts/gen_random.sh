#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --output=./slurm-outputs/%x-%j.out
#SBATCH --time=01-00:00:00     # DD-HH:MM:SS
#SBATCH --mail-user=marko.huang@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load python/3.7 cuda cudnn
module load nixpkgs/16.09  gcc/7.3.0
module load rdkit/2019.03.4

SOURCEDIR=`pwd`
TARGET=lumo
SAVE_DIR="${SLURM_JOB_ID}-${TARGET}"

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt
cd $SOURCEDIR/gen_random

python -u gen_random.py --target $TARGET --jobid $SLURM_JOB_ID > $SLURM_JOB_ID.out
python -u gen_random.py --target $TARGET --jobid $SLURM_JOB_ID >> $SLURM_JOB_ID.out
python -u gen_random.py --target $TARGET --jobid $SLURM_JOB_ID >> $SLURM_JOB_ID.out
python -u gen_random.py --target $TARGET --jobid $SLURM_JOB_ID >> $SLURM_JOB_ID.out
python -u gen_random.py --target $TARGET --jobid $SLURM_JOB_ID >> $SLURM_JOB_ID.out