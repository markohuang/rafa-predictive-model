#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle as pickle

from jtnn import *
import rdkit
from rdkit import RDLogger
import glob, shutil, os
from argparse import Namespace


# In[2]:


def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
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


# In[21]:


# all = pd.read_csv('data/oled-all/oled_all.csv')
# rafa = pd.read_csv('data/rafa/mols_rafadb.csv')
# l1 = np.array(rafa.smiles.tolist())
# l2 = np.array(all.smiles.tolist())
# res = np.intersect1d(l1, l2)
# with open('data/oled-all/all.txt', 'w') as f:
#     for item in l2:
#         f.write("%s\n" % item)


# In[23]:


lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)

opts = Namespace(**{
    'database': 'oled-all',
    'njobs': 8,
    'nsplits': 10
})

pool = Pool(opts.njobs)
num_splits = int(opts.nsplits)

with open(f'./data/{opts.database}/all.txt') as f:
    data = [line.strip("\r\n ").split()[0] for line in f]

all_data = pool.map(tensorize, data)

le = int((len(all_data) + num_splits - 1) / num_splits)

for split_id in range(num_splits):
    st = int(split_id * le)
    sub_data = all_data[st : st + le]

    with open('tensors-%d.pkl' % split_id, 'wb') as f:
        pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)


processed_folder = f'{opts.database}-processed'
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
for f in glob.glob(r'*-*.pkl'):
    shutil.move(f, os.path.join(processed_folder, f))

print('\n---Preprocessing Complete---\n')

