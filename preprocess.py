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

if __name__ == "__main__":
    lg = RDLogger.logger() 
    lg.setLevel(RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-n", "--split", dest="nsplits", default=5)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-t", "--target", dest="target", default="homo")
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)
    
    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    target = opts.target

    # for encoder target prediction
    train_labels = np.load('./data/rafa/' + target + '/train_labels.npy')
    val_labels = np.load('./data/rafa/' + target + '/val_labels.npy')
    test_labels = np.load('./data/rafa/' + target + '/test_labels.npy')
    
    with open('./data/rafa/' + target + '/train.txt') as f:
        data = [(line.strip("\r\n ").split()[0], label) for line, label in zip(f, train_labels)]

    with open('./data/rafa/' + target + '/val.txt') as f:
        val_data = [(line.strip("\r\n ").split()[0], label) for line, label in zip(f, val_labels)]

    with open('./data/rafa/' + target + '/test.txt') as f:
        test_data = [(line.strip("\r\n ").split()[0], label) for line, label in zip(f, test_labels)]

    train_data = pool.map(tensorize, data)
    val_data = pool.map(tensorize, val_data)
    test_data = pool.map(tensorize, test_data)

    package_data(train_data, num_splits, os.path.join(target, 'rafa-train'), 'train')
    package_data(val_data, num_splits, os.path.join(target, 'rafa-val'), 'val')
    package_data(test_data, num_splits, os.path.join(target, 'rafa-test'), 'test')
    print('\n---Preprocessing Complete---\n')
