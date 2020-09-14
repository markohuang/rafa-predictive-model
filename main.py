import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from jtnn import *
from auxiliaries import *
import rdkit


if __name__ == "__main__":
    import json, os
    from argparse import Namespace
    
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # get arguments
    cmd_args = vars(build_parser().parse_args())
    cmd_args = {k: v for k, v in cmd_args.items() if v is not None}
    json_path = cmd_args.get('load_json', './default_args.json')
    with open(json_path) as handle:
        args = json.loads(handle.read())
    args.update(cmd_args)
    if args.has_key('seed'):
        set_random_seed(args['seed'])
    else:
        args['seed'] = set_random_seed()
    if not args.has_key('save_dir'):
        args['save_dir'] = "{}-h{}-l{}-n{}-e{}-s{}".format(
            args['target'], args['hidden_size'], args['latent_size'],
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
    device = 'cuda' if args.cuda else 'cpu'
    train_path = os.path.join(args.target, 'rafa-train')
    val_path = os.path.join(args.target, 'rafa-val')
    print args
    

    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)

    model = RAFAVAE(vocab, args, evaluate=False)
    if args.cuda:
        model = model.cuda()
    print model

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if args.load_epoch > 0:
        model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

    print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    scheduler.step()

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = args.load_epoch
    meters = np.zeros(1)

    for epoch in xrange(args.epoch):
        loader = MolTreeFolder(train_path, vocab, args.batch_size, num_workers=4)
        val_loader = MolTreeFolder(val_path, vocab, args.batch_size, num_workers=4)
        for batch in loader:
            total_step += 1
            try:
                model.zero_grad()
                loss = model(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
            except Exception as e:
                print e
                continue

            meters = meters + np.array([loss.data.cpu().numpy()])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print "[%d] Loss: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, meters[0], param_norm(model), grad_norm(model))
                sys.stdout.flush()
                meters *= 0

            if total_step % args.save_iter == 0:
                torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                print "learning rate: %.6f" % scheduler.get_lr()[0]

        val_loss = 0.0
        val_steps = 0
        for val_batch in val_loader:
            val_steps += 1
            try:
                model.zero_grad()
                loss = model(val_batch)
                val_loss += float(loss.data)
            except Exception as e:
                print e
                continue

        print "Validation Loss: %.6f" % (val_loss / val_steps)
        sys.stdout.flush()

