from jtnn import *
import numpy as np
import torch
import argparse
from argparse import Namespace
import json, os, re


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--model_path', type=str, default='')
args = parser.parse_args()
model_path = args.model_path if args.model_path else False
model_dir = args.model_dir
json_path = os.path.join(model_dir, 'model.json')
with open(json_path) as handle:
    args = json.loads(handle.read())
args = Namespace(**args)

target = args.target
test_path = os.path.join(args.target, 'rafa-test')
# get last model if path not provided
if not model_path:
    models = os.listdir(model_dir)
    models = [x for x in models if re.search(r'model.iter-\d+', x)]
    model_path = os.path.join(model_dir, sorted(models, key=lambda x: int(x.split('-')[-1]))[-1])
device = 'cuda' if args.cuda else 'cpu'

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = RAFAVAE(vocab, args, evaluate=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
loader = MolTreeFolder(test_path, vocab, args.batch_size, num_workers=4)
pred = np.zeros(0)
act = np.zeros(0)
for batch in loader:
    output = model(batch)
    output = output.data.cpu().numpy().squeeze()
    labels = np.array([x.label for x in batch[0]])
    pred = np.concatenate((pred, output))
    act = np.concatenate((act, labels))


# plot
save_dir = model_dir
np.save(save_dir+'/pred-test', pred)
np.save(save_dir+'/act-test', act)

# save_dir = 'homo-h40-l56-n3-e20-s1481720974'
# target = 'homo'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
pred, act = np.load(save_dir+'/pred-test.npy'), np.load(save_dir+'/act-test.npy')
plt.figure()
ax = plt.gca()
ax.plot([0,1],[0,1], transform=ax.transAxes, color='green')
ax.set_aspect('equal')
plt.scatter(act, pred, s=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(target)
plt.savefig(save_dir+'/'+target+'-test.png', dpi=300)

