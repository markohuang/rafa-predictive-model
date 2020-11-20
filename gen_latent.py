import torch
import torch.nn as nn
from argparse import Namespace

import numpy as np  
from jtnn import *
from auxiliaries import load_args_and_vocab

from tqdm import tqdm
import rdkit
from rdkit import RDLogger
lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)

TARGET = 'lumo'
GEN_MODEL = 'gen-h450-l56-n3-e150-s3636887552'
# PRED_MODEL = 'ax-rate-12182694'
PRED_MODEL = 'ax-lumo-12157858'
# PRED_MODEL = 'ax-homo-12865653'
VOCAB_PATH = 'data/rafa/vocab.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# smiles = load_train_smiles() # this is for stuff like sascore
gmodel = gen_model(GEN_MODEL)
pmodel = pred_model(PRED_MODEL)

vocab = [x.strip("\r\n ") for x in open(VOCAB_PATH)] 
vocab = Vocab(vocab)

# prep data for input to model
loader = MolTreeFolder(f'{TARGET}/rafa-train', vocab, 32, num_workers=4)
scores = np.array([])
latent_points = []
for batch in tqdm(loader):
    pred_score = pmodel(batch).data.cpu().numpy().squeeze()
    scores = np.concatenate((scores, pred_score))
    smiles = [x.smiles for x in batch[0]]
    latent_points.append(gmodel.encode_latent_mean(smiles).data.cpu().numpy())

latent_points = np.vstack(latent_points)
np.savetxt(f'{TARGET}-latent_features.txt', latent_points)
np.savetxt(f'{TARGET}-scores.txt', np.array(scores))
