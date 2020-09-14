TARGET=homo
SOURCEDIR=`pwd`
SLURM_JOB_ID=local
export PYTHONPATH=$PYTHONPATH:$SOURCEDIR

cd $SOURCEDIR/data/rafa
# Setup data
python setup_data.py --target $TARGET

cd $SOURCEDIR/auxiliaries
# Get vocab
# python ../jtnn/mol_tree.py < ../data/rafa/all.txt > ../data/rafa/vocab.txt

# Preprocess
# python preprocess.py --target $TARGET --jobs 16

# Train
LATENT_SIZE=56
HIDDEN_SIZE=450
BATCH_SIZE=32
EPOCH=15
MODEL_DIR_PATH=model-$TARGET/$SLURM_JOB_ID-lsize$LATENT_SIZE-hsize$HIDDEN_SIZE-bsize$BATCH_SIZE-e$EPOCH/
mkdir -p $MODEL_DIR_PATH
python main.py \
--train $TARGET/rafa-train \
--val $TARGET/rafa-val \
--vocab ../data/rafa/vocab.txt \
--save_dir $MODEL_DIR_PATH \
--latent_size $LATENT_SIZE \
--hidden_size $HIDDEN_SIZE \
--batch_size $BATCH_SIZE \
--save_iter 500 \
--epoch $EPOCH > ${MODEL_DIR_PATH}output.txt

wait
# Test and plot
MODEL_PATH=$(ls ${MODEL_DIR_PATH}*[0-9] -1r | head -1)
python test_and_plot.py \
--vocab ../data/rafa/vocab.txt \
--target $TARGET \
--save_dir $MODEL_DIR_PATH \
--model $MODEL_PATH \
--latent_size $LATENT_SIZE \
--hidden_size $HIDDEN_SIZE \
--batch_size $BATCH_SIZE \
--data $TARGET/rafa-train

