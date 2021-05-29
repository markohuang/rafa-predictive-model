#!/usr/bin/env python
# coding: utf-8

# In[55]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import pandas as pd
import argparse
from argparse import Namespace
from collections import deque
import pickle as pickle
from tqdm import tqdm

from jtnn import *
from auxiliaries import build_parser, set_random_seed
from auxiliaries import set_random_seed, load_args_and_vocab
import rdkit
import json, os
from rdkit import RDLogger


# ## Train Set

# In[7]:


df = pd.read_csv('data/rafa/mols_rafadb.csv')
data = df.smiles


# ## Gen Model

# In[2]:


model_dir = 'gen-2dmodel-h450-l4-n3-e150-s256395117'
args, vocab = load_args_and_vocab(model_dir, 'data/rafa/vocab.txt')
args['cuda'] = 1
args = Namespace(**args)
gmodel = JTNNVAE(vocab, args)
gmodel.load_state_dict(torch.load(f'{model_dir}/model', map_location='cuda'))
gmodel.cuda();


# ## Pred Models

# In[39]:


def tensorize(payload):
    mol_tree = MolTree(payload)
    mol_tree.recover()
    del mol_tree.mol
    for node in mol_tree.nodes:
        node.cands = []
        del node.mol
    return mol_tree

def score_smile(x, pmodel, vocab=vocab):
    batch = tensorize(x)
    dataset = MolTreeDataset([[batch]], vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    with torch.no_grad():
        for b in dataloader: break
        score = pmodel(b).data.cpu().numpy().squeeze()
    del dataloader, dataset, batch
    return score


# In[3]:


model_dir = 'ax-splitting-19081910'
target = model_dir.split('-')[1]
args, vocab = load_args_and_vocab(model_dir, 'data/rafa/vocab.txt')
args = Namespace(**args)
pmodel = RAFAVAE(vocab, args, evaluate=True)
pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location='cuda'))
pmodel.cuda();
splitting = pmodel


# In[4]:


model_dir = 'ax-strength-13255643'
target = model_dir.split('-')[1]
args, vocab = load_args_and_vocab(model_dir, 'data/rafa/vocab.txt')
args = Namespace(**args)
pmodel = RAFAVAE(vocab, args, evaluate=True)
pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location='cuda'))
pmodel.cuda();
strength = pmodel


# In[5]:


model_dir = 'ax-lumo-12157858'
target = model_dir.split('-')[1]
args, vocab = load_args_and_vocab(model_dir, 'data/rafa/vocab.txt')
args = Namespace(**args)
pmodel = RAFAVAE(vocab, args, evaluate=True)
pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location='cuda'))
pmodel.cuda();
lumo = pmodel


# In[6]:


model_dir = 'ax-homo-12865654'
target = model_dir.split('-')[1]
args, vocab = load_args_and_vocab(model_dir, 'data/rafa/vocab.txt')
args = Namespace(**args)
pmodel = RAFAVAE(vocab, args, evaluate=True)
pmodel.load_state_dict(torch.load(f'{model_dir}/{args.target}-model', map_location='cuda'))
pmodel.cuda();
homo = pmodel


# ## Plot

# In[53]:


tree_vecs = []
mol_vecs = []

str_list = []
spl_list = []
homo_list = []
lumo_list = []


# In[56]:


for t in tqdm(data):
    latents = gmodel.encode_latent_mean([t]).detach().cpu().numpy()[0]
    tree_vecs.append(latents[:2])
    mol_vecs.append(latents[2:])
    str_list.append(score_smile(t, strength))
    spl_list.append(score_smile(t, splitting))
    homo_list.append(score_smile(t, homo))
    lumo_list.append(score_smile(t, lumo))


# In[58]:


import matplotlib.pyplot as plt


# In[59]:


tree_vecs = np.array(tree_vecs)
mol_vecs = np.array(mol_vecs)

str_list = np.array(str_list)
spl_list = np.array(spl_list)
homo_list = np.array(homo_list)
lumo_list = np.array(lumo_list)


# In[64]:


plt.scatter(tree_vecs[:,0],tree_vecs[:,1])

