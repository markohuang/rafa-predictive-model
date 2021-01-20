import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from argparse import Namespace
from collections import deque
import pickle as pickle

from jtnn import *
from auxiliaries import build_parser, set_random_seed
import rdkit
import json, os
from rdkit import RDLogger

lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)

# get arguments
cmd_args = vars(build_parser().parse_args())
cmd_args = {k: v for k, v in list(cmd_args.items()) if v is not None}
json_path = cmd_args.get('load_json', './default_gen_args.json')
with open(json_path) as handle:
    args = json.loads(handle.read())
args.update(cmd_args)
if 'seed' in args:
    set_random_seed(args['seed'])
else:
    args['seed'] = set_random_seed()
if 'save_dir' not in args:
    args['save_dir'] = "gen-{}-h{}-l{}-n{}-e{}-s{}".format(
        args['database'], args['hidden_size'], args['latent_size'],
        args['num_layers'], args['epoch'], args['seed']
    )
# save model settings
if not os.path.exists(args['save_dir']):
    os.makedirs(args['save_dir'])
dump_json_path = os.path.join(args['save_dir'], 'model.json')
if not os.path.exists(dump_json_path):
    with open(dump_json_path, "w") as fp:
        json.dump(args, fp, sort_keys=True, indent=4)
args = Namespace(**args)
print(args)
train_path = os.path.join(f'{args.database}-processed')
device = 'cuda' if args.cuda else 'cpu'


vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args)
if args.cuda:
    model = model.cuda()
print(model)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print(("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = args.beta
meters = np.zeros(4)

for epoch in range(args.epoch):
    print(f"Currently at epoch: {epoch+1}")
    loader = MolTreeFolder(train_path, vocab, args.batch_size, num_workers=4)
    for batch in loader:
        total_step += 1
        model.zero_grad()
        loss, kl_div, wacc, tacc, sacc = model(batch, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print(("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model))))
            sys.stdout.flush()
            meters *= 0

        if total_step % args.save_iter == 0:
            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print(("learning rate: %.6f" % scheduler.get_lr()[0]))

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)

torch.save(model.state_dict(), args.save_dir + f"/model")
