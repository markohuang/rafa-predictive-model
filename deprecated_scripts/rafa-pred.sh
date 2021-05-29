#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --output=./slurm-outputs/%x-%j.out
#SBATCH --time=23:59:59     # DD-HH:MM:SS

module load python/3.7 cuda cudnn
# module load nixpkgs/16.09  gcc/7.3.0
# module load rdkit/2019.03.4
module load gentoo/2020
module load gcc/9.3.0
module load rdkit/2020.09.1b1


TARGET=splitting
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

# Get vocab
# python ./jtnn/mol_tree.py < ./data/rafa/all.txt > ./data/rafa/vocab.txt

# Preprocess
# cd $SOURCEDIR/data/rafa
# python setup_data.py --target $TARGET
# cd $SOURCEDIR/
# python pred_preprocess.py --target $TARGET --jobs 16

python -u train_pred_main.py --save_dir ax-$TARGET-$SLURM_JOB_ID --target $TARGET --load_json ./$TARGET-default-model.json
python test_and_plot.py --model_dir ax-$TARGET-$SLURM_JOB_ID \
--model_path ax-$TARGET-$SLURM_JOB_ID/$TARGET-model
