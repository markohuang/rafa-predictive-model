# RAFA Property Predictive Model

Pulled out necessary components from Junction Tree Variational Autoencoder github. Added single dense layer for property prediction.

# Quick Start
* `default_args.json` contains default model parameters
* `local.sh` contains sample bash code
* `python main.py` trains the predictive model
* `python test_and_plot.py --model_dir` plots prediction v. actual on test set of the latest model in model_dir
