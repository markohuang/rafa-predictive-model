from jtnn import *
import numpy as np
import torch
import argparse
from argparse import Namespace
import json, os


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
args = parser.parse_args()
model_dir = args.model_dir
json_path = os.path.join(model_dir, 'model.json')
with open(json_path) as handle:
    args = json.loads(handle.read())
args = Namespace(**args)

save_dir = args.save_dir
target = args.target

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = RAFAVAE(vocab, args.hidden_size, args.latent_size, args.n_out, args.depthT, args.depthG, True)
model.load_state_dict(torch.load(args.model, map_location='cpu'))
model.eval()
loader = MolTreeFolder(args.data, vocab, args.batch_size, num_workers=4)
pred = np.zeros(0)
act = np.zeros(0)
for batch in loader:
    output = model(batch)
    output = output.data.numpy().squeeze()
    labels = np.array([x.label for x in batch[0]])
    pred = np.concatenate((pred, output))
    act = np.concatenate((act, labels))

np.save(save_dir+'/pred', pred)
np.save(save_dir+'/act', act)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
pred, act = np.load(save_dir+'/pred.npy'), np.load(save_dir+'/act.npy')
plt.figure()
ax = plt.gca()
ax.plot([0,1],[0,1], transform=ax.transAxes, color='green')
ax.set_aspect('equal')
plt.scatter(act, pred, s=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(target)
plt.savefig(save_dir+'/'+target+'.png', dpi=300)

