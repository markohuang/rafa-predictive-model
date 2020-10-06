__all__ = ['build_parser', 'set_random_seed', 'get_args_and_vocab', 'load_args_and_vocab']

import time
import torch
import logging
import json, os
import argparse
import numpy as np
from jtnn import Vocab


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--load_json', type=str)
    # required for predictive model
    parser.add_argument('--target', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--val', type=str)
    parser.add_argument('--vocab', type=str)
    # required for hyperparameter search
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--load_epoch', type=int)
    parser.add_argument('--total_trials', type=int)

    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--latent_size', type=int)
    parser.add_argument('--use_activation', type=bool)
    parser.add_argument('--n_out', type=int)
    parser.add_argument('--depthT', type=int)
    parser.add_argument('--depthG', type=int)
    parser.add_argument('--beta', type=float, default=0.0)

    parser.add_argument('--lr', type=float)
    parser.add_argument('--clip_norm', type=float)

    parser.add_argument('--epoch', type=int)
    parser.add_argument('--anneal_rate', type=float)
    parser.add_argument('--anneal_iter', type=int)
    parser.add_argument('--kl_anneal_iter', type=int)
    parser.add_argument('--print_iter', type=int)
    parser.add_argument('--save_iter', type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    return parser


def set_random_seed(seed=None):
    # 1) if seed not present, generate based on time
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
            ((seed & 0xFF000000) >> 24)
            + ((seed & 0x00FF0000) >> 8)
            + ((seed & 0x0000FF00) << 8)
            + ((seed & 0x000000FF) << 24)
        )
    # 2) Set seed for numpy (e.g. splitting)
    np.random.seed(seed)
    # 3) Set seed for torch (manual_seed now seeds all CUDA devices automatically)
    torch.manual_seed(seed)
    logging.info("Random state initialized with seed {:<10d}".format(seed))
    return seed


def get_args_and_vocab():
    # for hyperparameter optimization loop
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--target', type=str)
    model_dir = parser.parse_args().save_dir
    with open(os.path.join(model_dir, 'model.json')) as handle:
        args = json.loads(handle.read())
    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)
    return args, vocab


def load_args_and_vocab(model_dir):
    with open(os.path.join(model_dir, 'model.json')) as handle:
        args = json.loads(handle.read())
    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)
    return args, vocab