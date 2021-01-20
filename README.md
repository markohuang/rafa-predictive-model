# RAFA Property Predictive Model

Pulled out necessary components from Junction Tree Variational Autoencoder github. Added single dense layer for property prediction.

# Quick Start
* `default_gen_args.json` contains default generative model parameters
* `default_pred_args.json` contains default predictive model parameters
* `rafa-gen.sh` contains sample bash code for training the jtnn generative model
* `rafa-pred.sh` contains sample bash code for training the mpn prediction model
* `python test_and_plot.py --model_dir` plots prediction v. actual on test set of the latest model in model_dir
