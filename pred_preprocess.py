#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
root = str(pathlib.Path().absolute())


# ## Setup Data

# In[2]:


import numpy as np
import pandas as pd
import argparse
import os
from argparse import Namespace


# In[3]:


args = Namespace(**{
    'target': 'strength',
    'database': 'oled-all',
    'csv_file': 'oled_all.csv'
})


# In[4]:


datapath = os.path.join(root, 'data', args.database, args.csv_file)
savepath = os.path.join(root, 'pred_processed', args.target)
os.makedirs(savepath, exist_ok=True)


# In[5]:


df = pd.read_csv(datapath)
df = df.dropna(subset=[args.target])
df = df[['smiles', args.target]]


# In[6]:


# clip outliers for rate
if args.target == 'rate':
    rate = df['rate']
    df = df[((rate > 1e-100) & (rate < 1e-8))]
    df['rate'] = np.log(df['rate'])
# if args.target == 'strength':
#     strength = df['strength']
#     df = df[strength > 0]


# In[7]:


# Three way split of 70% train set, 20% val set, 10% test set
train, val, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.9*len(df))])
# All
x0 = df['smiles']
y0 = df[args.target]
x0.to_csv(os.path.join(savepath, 'all.txt'), header=None, index=None, sep=' ')
np.save(os.path.join(savepath, 'all_labels'), np.array(y0.values))
# Train
x1 = train['smiles']
y1 = train[args.target]
x1.to_csv(os.path.join(savepath, 'train.txt'), header=None, index=None, sep=' ')
np.save(os.path.join(savepath, 'train_labels'), np.array(y1.values))
# Val
x2 = val['smiles']
y2 = val[args.target]
x2.to_csv(os.path.join(savepath, 'val.txt'), header=None, index=None, sep=' ')
np.save(os.path.join(savepath, 'val_labels'), np.array(y2.values))
# Test
x3 = test['smiles']
y3 = test[args.target]
x3.to_csv(os.path.join(savepath, 'test.txt'), header=None, index=None, sep=' ')
np.save(os.path.join(savepath, 'test_labels'), np.array(y3.values))


# ## Preprocessing

# In[8]:


import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle as pickle

from jtnn import *
import rdkit
from rdkit import RDLogger
import glob, shutil


# In[9]:


opts = Namespace(**{**{
    'njobs': 8,
    'nsplits': 10
}, **vars(args)})


# In[10]:


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

def package_data(data, num_splits, processed_folder, prefix):
    le = int((len(data) + num_splits - 1) / num_splits)
    for split_id in range(num_splits):
        st = int(split_id * le)
        sub_data = data[st : st + le]
        with open('{}-tensors-{}.pkl'.format(prefix, split_id), 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    for f in glob.glob(r'*-*-*.pkl'):
        shutil.move(f, os.path.join(processed_folder, f))


# In[11]:


lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)

pool = Pool(opts.njobs)
num_splits = int(opts.nsplits)
target = opts.target


# In[12]:


# for encoder target prediction
train_labels = np.load(os.path.join(savepath, 'train_labels.npy'))
val_labels = np.load(os.path.join(savepath, 'val_labels.npy'))
test_labels = np.load(os.path.join(savepath, 'test_labels.npy'))
with open(os.path.join(savepath, 'train.txt')) as f:
    data = [(line.strip("\r\n ").split()[0], label) for line, label in zip(f, train_labels)]
with open(os.path.join(savepath, 'val.txt') )as f:
    val_data = [(line.strip("\r\n ").split()[0], label) for line, label in zip(f, val_labels)]
with open(os.path.join(savepath, 'test.txt')) as f:
    test_data = [(line.strip("\r\n ").split()[0], label) for line, label in zip(f, test_labels)]


# In[ ]:


# train_data = pool.map(tensorize, data)
# val_data = pool.map(tensorize, val_data)
# test_data = pool.map(tensorize, test_data)
train_data = list(map(tensorize, data))
val_data = list(map(tensorize, val_data))
test_data = list(map(tensorize, test_data))


# In[14]:


package_data(train_data, num_splits, os.path.join(savepath, 'train'), 'train')
package_data(val_data, num_splits, os.path.join(savepath, 'val'), 'val')
package_data(test_data, num_splits, os.path.join(savepath, 'test'), 'test')
print('\n---Preprocessing Complete---\n')

