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
from auxiliaries import build_parser, set_random_seed, get_args_and_vocab
import rdkit
import json, os
from rdkit import RDLogger



def trainer(model, args, vocab):
    train_path = os.path.join(args.target, 'rafa-train')
    val_path = os.path.join(args.target, 'rafa-val')

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if args.load_epoch > 0:
        model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

    print(("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate, verbose=True)
    # As per warning
    # scheduler.step()

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = args.load_epoch
    meters = np.zeros(1)

    for epoch in range(args.epoch):
        loader = MolTreeFolder(train_path, vocab, args.batch_size, num_workers=4)
        val_loader = MolTreeFolder(val_path, vocab, args.batch_size, num_workers=4)
        model.train()
        for batch in loader:
            total_step += 1
            model.zero_grad()
            loss = model(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters = meters + np.array([loss.data.cpu().numpy()])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(("[%d] Loss: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, meters[0], param_norm(model), grad_norm(model))))
                sys.stdout.flush()
                meters *= 0

            if total_step % args.save_iter == 0:
                torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                print(("learning rate: %.6f" % scheduler.get_last_lr()[0]))

        val_loss = 0.0
        val_steps = 0
        model.eval()
        for val_batch in val_loader:
            val_steps += 1
            loss = model(val_batch)
            val_loss += float(loss.data)

        print(("Validation Loss: %.6f" % (val_loss / val_steps)))
        sys.stdout.flush()
    return model


def train(parametrization):
    args, vocab = get_args_and_vocab()
    args = {**args, **parametrization}
    args = Namespace(**args)
    print(args)
    model = RAFAVAE(vocab, args, evaluate=False)
    if args.cuda:
        model = model.cuda()
    return trainer(model, args, vocab)


def evaluate(model):
    args, vocab = get_args_and_vocab()
    args = Namespace(**args)
    test_path = os.path.join(args.target, 'rafa-test')
    model.eval()
    loader = MolTreeFolder(test_path, vocab, args.batch_size, num_workers=4)
    iters = 0
    loss = 0
    for batch in loader:
        loss += float(model(batch).data)
        iters += 1
    return loss / iters


def plot_train(model):
    args, vocab = get_args_and_vocab()
    args = Namespace(**args)
    train_path = os.path.join(args.target, 'rafa-train')
    model.set_mode(evaluate=True)
    model.eval()
    loader = MolTreeFolder(train_path, vocab, args.batch_size, num_workers=4)
    pred = np.zeros(0)
    act = np.zeros(0)
    for batch in loader:
        output = model(batch)
        output = output.data.cpu().numpy().squeeze()
        labels = np.array([x.label for x in batch[0]])
        pred = np.concatenate((pred, output))
        act = np.concatenate((act, labels))
    save_dir = args.save_dir
    np.save(save_dir+'/pred-train', pred)
    np.save(save_dir+'/act-train', act)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred, act = np.load(save_dir+'/pred-train.npy'), np.load(save_dir+'/act-train.npy')
    plt.figure()
    ax = plt.gca()
    ax.plot([0,1],[0,1], transform=ax.transAxes, color='green')
    ax.set_aspect('equal')
    plt.scatter(act, pred, s=1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(args.target)
    plt.savefig(save_dir+'/'+args.target+'-train.png', dpi=300)


def train_evaluate(parametrization):
    model = train(parametrization)
    return evaluate(model)


if __name__ == "__main__":
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # get arguments
    cmd_args = vars(build_parser().parse_args())
    cmd_args = {k: v for k, v in list(cmd_args.items()) if v is not None}
    json_path = cmd_args.get('load_json', './default_pred_args.json')
    with open(json_path) as handle:
        args = json.loads(handle.read())
    args.update(cmd_args)
    if 'seed' in args:
        set_random_seed(args['seed'])
    else:
        args['seed'] = set_random_seed()
    if 'save_dir' not in args:
        args['save_dir'] = "{}-h{}-l{}-n{}-e{}-s{}".format(
            args['target'], args['hidden_size'], args['latent_size'],
            args['num_layers'], args['epoch'], args['seed']
        )
    # save model settings
    os.makedirs(args['save_dir'], exist_ok=True)
    dump_json_path = os.path.join(args['save_dir'], 'model.json')
    if not os.path.exists(dump_json_path):
        with open(dump_json_path, "w") as fp:
            json.dump(args, fp, sort_keys=True, indent=4)
    args = Namespace(**args)
    device = 'cuda' if args.cuda else 'cpu'
    print(args)

    # hyperparameter search
    from ax.service.managed_loop import optimize
    best_parameters, values, experiment, model = optimize(
        parameters = [
            { "name": "lr", "type": "range", "bounds": [1e-5, 5e-3] },
            # { "name": "hidden_size", "type": "range", "value_type": "int", "bounds": [100, 650] },
            # { "name": "num_layers", "type": "range", "value_type": "int", "bounds": [1, 5] },
            # { "name": "latent_size", "type": "range", "value_type": "int", "bounds": [40, 100] },
            { "name": "epoch", "type": "range", "value_type": "int", "bounds": [80, 150] },
        ],
        evaluation_function=train_evaluate,
        minimize=True,
        total_trials=args.total_trials
    )
    means, covariances = values
    print('best parameters:', best_parameters)
    print(means)

    # re-train best model
    best_parameters['plot'] = True
    model = train(best_parameters)
    torch.save(model.state_dict(), args.save_dir + f"/{args.target}-model")
    best_model_param = { **vars(args), **best_parameters }
    with open(dump_json_path, "w") as fp:
        json.dump(best_model_param, fp, sort_keys=True, indent=4)
    
    # plot pred v. act for train set
    plot_train(model)

    # log best_parameters and objective separately
    ilogpath = os.path.join(args.save_dir, 'model.json')
    with open(ilogpath, 'w') as f:
        log = 'Best parameters:\n'
        log += '\n'.join(f'{u:<11}: {v}' for u, v in best_parameters.items())
        log += f"\n\nObjective: {means['objective']}"
        f.write(log)


