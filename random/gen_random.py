if __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from jtnn import *
from auxiliaries import set_random_seed, load_args_and_vocab
from argparse import Namespace

import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import argparse

import rdkit
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--target', type=str, default='lumo')
parser.add_argument('--pred_model', type=str, default='../ax-lumo-12157858')
parser.add_argument('--gen_model', type=str, default='../gen-h450-l56-n3-e150-s3636887552')
args = parser.parse_args()

# parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
dtype = torch.double
TARGET = args.target
D = 56  # latent_size
X = np.loadtxt(f'../latent_features.txt')
min_x, max_x = stats.describe(X).minmax
BOUNDS = torch.tensor([min_x, max_x], device=DEVICE, dtype=dtype)
PRED_MODEL = args.pred_model
GEN_MODEL = args.gen_model
# set random seed
seed = set_random_seed()
# seed = 4079334741
print(f'random seed: {seed}')
torch.manual_seed(seed)

def unnormalize(X: Tensor, bounds: Tensor) -> Tensor:
    return X * (bounds[1] - bounds[0]) + bounds[0]


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


def scorer():
    model_dir = PRED_MODEL
    args, vocab = load_args_and_vocab(model_dir, '../data/rafa/vocab.txt')
    args['cuda'] = 0 if DEVICE == 'cpu' else 1
    args = Namespace(**args)
    pmodel = RAFAVAE(vocab, args, evaluate=True)
    pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location=DEVICE))
    pmodel.to(DEVICE)
    return pmodel


def score_smile(x, pmodel, vocab=vocab):
    batch = tensorize(x)
    dataset = MolTreeDataset([[batch]], vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    with torch.no_grad():
        for b in dataloader: break
        score = pmodel(b).data.cpu().numpy().squeeze()
    return score


def decoder():
    model_dir = GEN_MODEL
    args, vocab = load_args_and_vocab(model_dir, '../data/rafa/vocab.txt')
    args['cuda'] = 0 if DEVICE == 'cpu' else 1
    args = Namespace(**args)
    gmodel = JTNNVAE(vocab, args)
    gmodel.load_state_dict(torch.load(f'{model_dir}/model', map_location=DEVICE))
    gmodel.to(DEVICE)
    return gmodel, vocab


def decode(X, gmodel, vocab=vocab):
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


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    df = pd.read_csv('../data/rafa/mols_rafadb.csv')
    NUM_SAMPLES = 100
    N_EPOCHS = 25
    K = 10
    TOT_SAMPLED = 0
    NUM_IN_TRAIN_DATA = 0
    gmodel, vocab = decoder()
    pmodel = scorer()
    bot_smiles, top_smiles = np.empty(0), np.empty(0)
    bot_smiles_score, top_smiles_score = np.empty(0), np.empty(0)
    for i in tqdm(range(N_EPOCHS)):
        new_x = unnormalize(torch.rand(NUM_SAMPLES, D, device=DEVICE, dtype=dtype), bounds=BOUNDS).cpu().numpy()
        print('...decoding...')
        smiles = decode(new_x, gmodel)
        smiles_score = []
        print('....scoring...')
        for x in smiles:
            try:
                smiles_score.append(score_smile(x, pmodel, vocab))
            except KeyError as err:
                print(f'{err} not in vocab')
        smiles_score = np.array(smiles_score)
        smiles = np.array(smiles)
        ind_botK, ind_topK = np.argpartition(smiles_score, K)[:K], np.argpartition(smiles_score, -K)[-K:]
        botK, topK = smiles_score[ind_botK],  smiles_score[ind_topK]
        # add top/bottom k to list
        bot_smiles = np.concatenate((bot_smiles, smiles[ind_botK]))
        top_smiles = np.concatenate((top_smiles, smiles[ind_topK]))
        bot_smiles_score = np.concatenate((bot_smiles_score, botK))
        top_smiles_score = np.concatenate((top_smiles_score, topK))

        TOT_SAMPLED += NUM_SAMPLES
        NUM_IN_TRAIN_DATA += (df.smiles.isin(smiles)).sum()

    np.save(f'./{seed}-top_smiles', top_smiles)
    np.save(f'./{seed}-bot_smiles', bot_smiles)
    np.save(f'./{seed}-top_smiles_score', top_smiles_score)
    np.save(f'./{seed}-bot_smiles_score', bot_smiles_score)

    print(f'Total sampled: {TOT_SAMPLED}, with {NUM_IN_TRAIN_DATA} in rafadb.')

    ind_botK, ind_topK = np.argpartition(bot_smiles_score, K)[:K], np.argpartition(top_smiles_score, -K)[-K:]
    botK, topK = bot_smiles_score[ind_botK], top_smiles_score[ind_topK]
    # bot_smiles[ind_botK], top_smiles[ind_topK]
    print('Lowest scores:')
    print('\n'.join(f'score: {x: 7.4f}, smiles: {y}' for x, y in zip(botK, bot_smiles[ind_botK])))
    print('Highest scores:')
    print('\n'.join(f'score: {x: 7.4f}, smiles: {y}' for x, y in zip(topK, top_smiles[ind_topK])))
    # [x for _, x in sorted(zip(top_smiles_score, top_smiles), key=lambda pair: pair[0])]
        