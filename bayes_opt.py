import torch
import torch.nn as nn
from jtnn import *
from auxiliaries import set_random_seed, load_args_and_vocab

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt


# parameters
D = 20
BOUNDS = torch.tensor([[-6.0] * D, [6.0] * D], device=device, dtype=dtype)
N_BATCH = 50
MC_SAMPLES = 2000
BATCH_SIZE = 3
PRED_MODEL = ''
GEN_MODEL = ''
DEVICE = 'cuda' if args.cuda else 'cpu'
# set random seed
seed = set_random_seed()
print(f'random seed: {seed}')
torch.manual_seed(seed)


def tensorize(payload, assm=True):
    mol_tree = MolTree(payload)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    return mol_tree


def score_smile(x):
    model_dir = f'pred_model/{PRED_MODEL}'
    args, vocab = load_args_and_vocab(model_dir)
    args = Namespace(**args)
    pmodel = RAFAVAE(vocab, args, evaluate=True)
    pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location=DEVICE))
    batch = map(tensorize, x)
    batches = [batch]
    dataset = MolTreeDataset(batches, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    with torch.no_grad():
        for b in dataloader: break
        scores = pmodel(b).data.cpu().numpy().squeeze()
    return scores


def decode(x):
    model_dir = f'gen_model/{GEN_MODEL}'
    args, vocab = load_args_and_vocab(model_dir)
    gmodel = JTNNVAE(vocab, args)
    gmodel.load_state_dict(torch.load(f'{model_dir}/model', map_location=DEVICE))
    tree_vec, mol_vec = x
    with torch.no_grad():
        smiles = gmodel.decode(tree_vec, mol_vec, prob_decode=False)
    return smiles


def gen_initial_data(n=5):
    # generate training data  
    train_x = unnormalize(torch.rand(n, D, device=device, dtype=dtype), bounds=BOUNDS)
    train_obj = score_smile(decode(train_x)).unsqueeze(-1)  # add output dimension
    best_observed_value = train_obj.max().item()
    return train_x, train_obj, best_observed_value


def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_model(mll)
    return model


def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.zeros(D, dtype=dtype, device=device), 
            torch.ones(D, dtype=dtype, device=device),
        ]),
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=200,
    )
    # observe new values 
    new_x = unnormalize(candidates.detach(), bounds=BOUNDS)
    new_obj = score_smile(decode(new_x)).unsqueeze(-1)  # add output dimension
    return new_x, new_obj

best_observed = []

# call helper function to initialize model
train_x, train_obj, best_value = gen_initial_data(n=5)
best_observed.append(best_value)


# run bayesian optimization loop
print(f"\nRunning BO ", end='')
state_dict = None
# run N_BATCH rounds of BayesOpt after the initial random batch
for iteration in range(N_BATCH):    

    # fit the model
    model = get_fitted_model(
        normalize(train_x, bounds=BOUNDS), 
        standardize(train_obj), 
        state_dict=state_dict,
    )
    
    # define the qNEI acquisition module using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
    qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max())

    # optimize and get new observation
    new_x, new_obj = optimize_acqf_and_get_observation(qEI)

    # update training points
    train_x = torch.cat((train_x, new_x))
    train_obj = torch.cat((train_obj, new_obj))

    # update progress
    best_value = train_obj.max().item()
    best_observed.append(best_value)
    
    state_dict = model.state_dict()
    
    print(".", end='')