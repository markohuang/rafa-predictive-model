# rafa-predictive-model

gen_models contains all generative models

pred_models contains predictive models for lumo, homo, rate, splitting, and strength

train_gen.sh: script for training a generative model, set latent size = 4 in train_gen.ipynb for 2d model. need to run gen_preprocess.sh first to setup data

train_pred: script for training a predictive model, set target in train_pred.ipynb. need to run pred_preprocess.sh first to setup data

misc.sh: job script to update vocab, needed to run 400k models. if error occurs need to remove the '.' from .vocab, .chemutils import statements in jtnn/mol_tree.py and jtnn/chemutils.py. re-add the dots after vocab is updated


