#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --output=./slurm-outputs/%x-%j.out
#SBATCH --time=02:00:00     # DD-HH:MM:SS

module load python/3.7 cuda cudnn
module load nixpkgs/16.09  gcc/7.3.0
module load rdkit/2019.03.4

TARGET=lumo
SOURCEDIR=`pwd`

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt
cd $SOURCEDIR/gpytorch
pip install .
cd $SOURCEDIR/botorch-0.3.0
pip install .
cd $SOURCEDIR/Ax-0.1.13
pip install .
cd $SOURCEDIR/

python gen_latent.py --target $TARGET --pred_model ax-$TARGET-13255643
# python bayes_opt.py --target lumo --pred_model ax-lumo-13304750
python bayes_opt.py --target $TARGET --pred_model ax-$TARGET-13255643
