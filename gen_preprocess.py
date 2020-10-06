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

if __name__ == "__main__":
    lg = RDLogger.logger() 
    lg.setLevel(RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open('./data/rafa/all.txt') as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    all_data = pool.map(tensorize, data)

    le = int((len(all_data) + num_splits - 1) / num_splits)

    for split_id in range(num_splits):
        st = int(split_id * le)
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
    
    
    processed_folder = 'rafa-processed'
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    for f in glob.glob(r'*-*.pkl'):
        shutil.move(f, os.path.join(processed_folder, f))

    print('\n---Preprocessing Complete---\n')

