import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from jtnn import *
from auxiliaries import set_random_seed, load_args_and_vocab
from argparse import Namespace

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import rdkit
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
dtype = torch.double
N_EPOCHS = 8
MC_SAMPLES = 2000
NUM_CANDIDATES = 60
LEARNING_RATE = 0.0075
MAX_ITER = 30
TARGET = 'lumo'
D = 56  # latent_size
X = np.loadtxt(f'{TARGET}-latent_features.txt')
min_x, max_x = stats.describe(X).minmax
BOUNDS = torch.tensor([min_x*0.8, max_x*1.2], device=DEVICE, dtype=dtype)
# PRED_MODEL = 'ax-rate-12182694'
PRED_MODEL = 'ax-lumo-12157858'
# PRED_MODEL = 'ax-homo-12865653'
GEN_MODEL = 'gen-h450-l56-n3-e150-s3636887552'
# set random seed
seed = set_random_seed()
# seed = 4079334741
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
    model_dir = PRED_MODEL
    args, vocab = load_args_and_vocab(model_dir)
    args['cuda'] = 0 if DEVICE == 'cpu' else 1
    args = Namespace(**args)
    pmodel = RAFAVAE(vocab, args, evaluate=True)
    pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location=DEVICE))
    pmodel.to(DEVICE)
    batch = tensorize(x)
    dataset = MolTreeDataset([[batch]], vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    with torch.no_grad():
        for b in dataloader: break
        score = pmodel(b).data.cpu().numpy().squeeze()
    return score


def decode(X):
    model_dir = GEN_MODEL
    args, vocab = load_args_and_vocab(model_dir)
    args['cuda'] = 0 if DEVICE == 'cpu' else 1
    args = Namespace(**args)
    gmodel = JTNNVAE(vocab, args)
    gmodel.load_state_dict(torch.load(f'{model_dir}/model', map_location=DEVICE))
    gmodel.to(DEVICE)
    smiles = []
    for i in range(X.shape[0]):
        x = X[i]
        x = x.reshape((1, -1))
        tree_vec, mol_vec = np.hsplit(x, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        with torch.no_grad():
            smiles.append(gmodel.decode(tree_vec, mol_vec, prob_decode=False))
    return smiles


def get_initial_data(n=5):
    # generate training data 
    # train_x = unnormalize(torch.rand(n, D, device=DEVICE, dtype=dtype), bounds=BOUNDS).cpu().numpy()
    # train_obj = np.array([score_smile(x) for x in decode(train_x)])[:, np.newaxis]  # add output dimension
    train_x = np.loadtxt(f'{TARGET}-latent_features.txt')
    train_obj = np.loadtxt(f'{TARGET}-scores.txt')[:, np.newaxis] # add output dimension
    best_observed_value = train_obj.max().item()
    return torch.tensor(train_x, device=DEVICE, dtype=dtype), torch.tensor(train_obj, device=DEVICE, dtype=dtype), best_observed_value


def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    # fit_gpytorch_model(mll)
    fit_gpytorch_torch(mll, options={'maxiter': MAX_ITER, 'lr': 0.0075})
    # DO ADAM HERE
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # model.train()
    # for epoch in range(MAX_ITER):
    #     # clear gradients
    #     optimizer.zero_grad()
    #     # forward pass through the model to obtain the output MultivariateNormal
    #     output = model(train_x)
    #     # Compute negative marginal log likelihood
    #     loss = - mll(output, train_obj).sum()
    #     # back prop gradients
    #     loss.backward()
    #     # print every 10 iterations
    #     print(f'\nloss.item is {loss.item()}')
    #     if (epoch + 1) % 10 == 0:
    #         print(
    #             f"Epoch {epoch+1:>3}/{MAX_ITER} - Loss: {loss.item():>4.3f} "
    #             f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():>4.3f} " 
    #             f"noise: {model.likelihood.noise.item():>4.3f}" 
    #         )
    #     optimizer.step()
    # model.eval()
    return model


def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.zeros(D, dtype=dtype, device=DEVICE), 
            torch.ones(D, dtype=dtype, device=DEVICE),
        ]),
        q=NUM_CANDIDATES,
        num_restarts=10,
        raw_samples=200,
    )
    # observe new values 
    new_x = unnormalize(candidates.detach(), bounds=BOUNDS).cpu().numpy()
    new_obj = np.array([score_smile(x) for x in decode(new_x)])[:, np.newaxis]
    return torch.tensor(new_x).to(DEVICE), torch.tensor(new_obj).to(DEVICE)

best_observed = []

# call helper function to initialize model
train_x, train_obj, best_value = get_initial_data(n=5)
best_observed.append(best_value)


# run bayesian optimization loop
print(f"\nRunning BO ")
state_dict = None
# run N_EPOCHS rounds of BayesOpt after the initial random batch
for iteration in range(N_EPOCHS):    
    print('-'*16, f'iteration {iteration+1}', '-'*16)
    # fit the model
    model = get_fitted_model(
        normalize(train_x, bounds=BOUNDS), 
        standardize(train_obj), 
        state_dict=state_dict,
    )
    
    # define the qNEI acquisition module using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
    qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max())
    # qEI = qNoisyExpectedImprovement(model=model, X_baseline=train_x, sampler=qmc_sampler, prune_baseline=True)

    # optimize and get new observation
    try:
        # for some reason generated smiles can have words not in vocab
        new_x, new_obj = optimize_acqf_and_get_observation(qEI)
        # update training points
        train_x = torch.cat((train_x, new_x))
        train_obj = torch.cat((train_obj, new_obj))

        # update progress
        best_value = train_obj.max().item()
        best_observed.append(best_value)
        
        state_dict = model.state_dict()
        
        # print(".", end='')
    except KeyError as err:
        print(f'{err} not in vocab')


# _, idx = np.unique(train_obj, return_index=True)
# unique_obj, unique_x = train_obj[idx], train_x[idx]
print(len(train_x))
_, idx = train_obj.squeeze().topk(40)
topk_obj = train_obj[idx].cpu()
topk_x = train_x[idx].cpu()

topsmiles = decode(topk_x.numpy())
_, idx = np.unique(np.array(topsmiles), return_index=True)
topsmiles = np.array(topsmiles)[idx]
topk_x = topk_x[idx].numpy().squeeze()
topk_obj = topk_obj[idx].numpy().squeeze()

with open(f'{TARGET}-best_smiles.txt', 'w') as f:
    f.write('\n'.join(topsmiles))

with open(f'{TARGET}-best_scores.txt', 'w') as f:
    f.write('\n'.join(str(x) for x in topk_obj))

# print('\n'.join(topsmiles))

print('best observed value is:')
print(best_observed)


from rdkit import Chem
from rdkit.Chem import Draw

print(topsmiles)
suppl = Chem.SmilesMolSupplier(f'{TARGET}-best_smiles.txt', titleLine=False)
ms = [x for x in suppl if x is not None]
svg = Draw.MolsToGridImage(
    ms,
    molsPerRow=5,
    subImgSize=(400, 400),
    legends=[f'{score:.2f}: {smiles}' for score, smiles in zip(topk_obj, topsmiles)],
    useSVG=True
)

with open(f'{TARGET}-best_smiles.svg', 'w') as f:
    f.write(svg)
