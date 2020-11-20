import pickle
import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os, os.path
import sascorer
import networkx as nx

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from jtnn import *

import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import RDLogger
os.environ['KMP_DUPLICATE_LIB_OK']='True'
lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)

TARGET = 'lumo'
GEN_MODEL = 'gen-h450-l56-n3-e150-s3636887552'
PRED_MODEL = 'ax-lumo-12157858'
VOCAB_PATH = 'data/rafa/vocab.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

def load_train_smiles():
    with open(f'data/rafa/{TARGET}/train.txt') as f:
        smiles = f.readlines()
    for i in range(len(smiles)):
        smiles[ i ] = smiles[ i ].strip()
    return smiles

def pred_model(model_dir):
    args, vocab = load_args_and_vocab(model_dir)
    args = Namespace(**args)
    pmodel = RAFAVAE(vocab, args, evaluate=True)
    pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location=DEVICE))
    pmodel.to(device=DEVICE)
    return pmodel

def gen_model(model_dir):
    args, vocab = load_args_and_vocab(model_dir)
    args = Namespace(**args)
    gmodel = JTNNVAE(vocab, args)
    gmodel.load_state_dict(torch.load(f'{model_dir}/model', map_location=DEVICE))
    gmodel.to(DEVICE)
    return gmodel

gmodel = gen_model(GEN_MODEL)
pmodel = pred_model(PRED_MODEL)

vocab = [x.strip("\r\n ") for x in open(VOCAB_PATH)] 
vocab = Vocab(vocab)

# We load the random seed
np.random.seed(random_seed)

# We load the data (y is minued!)
X_train = np.loadtxt('latent_features.txt')
y = np.loadtxt(f'{TARGET}-scores.txt')
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]


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

iteration = 0
while iteration < 4:
    # We fit the GP
    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 25, learning_rate = 0.0075)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print 'Test RMSE: ', error
    print 'Test ll: ', testll

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print 'Train RMSE: ', error
    print 'Train ll: ', trainll

    # We pick the next 60 inputs
    next_inputs = sgp.batched_greedy_ei(60, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in xrange(60):
        all_vec = next_inputs[i].reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = gmodel.decode(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            new_features.append(all_vec)
    
    print len(valid_smiles), "molecules are found"
    valid_smiles = valid_smiles[:50]
    new_features = next_inputs[:50]
    new_features = np.vstack(new_features)
    save_object(valid_smiles, opts.save_dir + "/valid_smiles{}.dat".format(iteration))

    batch = map(tensorize, valid_smiles)
    batches = [batch]
    dataset = MolTreeDataset(batches, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    for b in dataloader: break
    scores = pmodel(b).data.cpu().numpy().squeeze()

    # scores = []
    # for i in range(len(valid_smiles)):
    #     current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles[ i ]))
    #     score = current_SA_score
    #     scores.append(-score) #target is always minused

    print valid_smiles
    print scores 

    save_object(scores, opts.save_dir + "/scores{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1
