#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --output=./slurm-outputs/%x-%j.out
#SBATCH --time=4:59:59     # DD-HH:MM:SS

module load python/3.7 cuda cudnn
module load nixpkgs/16.09  gcc/7.3.0
module load rdkit/2019.03.4

SOURCEDIR=`pwd`

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt
cd $SOURCEDIR/


## RAFA
# Get vocab
# python ./jtnn/mol_tree.py < ./data/rafa/all.txt > ./data/rafa/vocab.txt
# Preprocess
# cd $SOURCEDIR/data/rafa
# python setup_data.py --target $TARGET
# cd $SOURCEDIR/
# python preprocess.py --target $TARGET --jobs 16

## POLYMERS
# Get vocab
# python ./jtnn/mol_tree.py < ./data/polymers/all.txt > ./data/polymers/vocab.txt
# Preprocess
cd $SOURCEDIR/
python gen_preprocess.py --database polymers --jobs 16
echo ---DONE---

# python train_gen_main.py --database polymers
