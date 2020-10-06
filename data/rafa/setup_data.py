import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='homo')
args = parser.parse_args()

df = pd.read_csv('mols_rafadb.csv')
df = df.dropna(subset=[args.target])
df = df[['smiles', args.target]]
# clip outliers for rate
if args.target == 'rate':
    df = df[((rate > 1e-100) & (rate < 1e-8))]

if not os.path.exists(args.target):
    os.makedirs(args.target)

# Three way split of 60% train set, 20% val set, 20% test set
train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
# All
x0 = df['smiles']
y0 = df[args.target]
x0.to_csv(os.path.join(args.target, 'all.txt'), header=None, index=None, sep=' ', mode='a')
np.save(os.path.join(args.target, 'all_labels'), np.array(y0.values))
# Train
x1 = train['smiles']
y1 = train[args.target]
x1.to_csv(os.path.join(args.target, 'train.txt'), header=None, index=None, sep=' ', mode='a')
np.save(os.path.join(args.target, 'train_labels'), np.array(y1.values))
# Val
x2 = val['smiles']
y2 = val[args.target]
x2.to_csv(os.path.join(args.target, 'val.txt'), header=None, index=None, sep=' ', mode='a')
np.save(os.path.join(args.target, 'val_labels'), np.array(y2.values))
# Test
x3 = test['smiles']
y3 = test[args.target]
x3.to_csv(os.path.join(args.target, 'test.txt'), header=None, index=None, sep=' ', mode='a')
np.save(os.path.join(args.target, 'test_labels'), np.array(y3.values))
