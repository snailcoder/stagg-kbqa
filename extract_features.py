#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse

import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Extract features for ranking and save features in current directory.')
parser.add_argument('type', choices=['patchain', 'qep'], help='features to extract')
parser.add_argument('infer_chain_model', help='infer chain model path')
parser.add_argument('infer_chain_vocab', help='infer chain vocab path')
parser.add_argument('infer_chain_data',
                    help='TSV file to train infer chain model, its 2nd and 3rd column '
                         'must be question (or pattern) and the predicate sequence')
# parser.add_argument('--rank_model', default='models/rank_best_model.pth', help='ranking model path')

args = parser.parse_args()

infer_chain_model = torch.load(args.infer_chain_model)
infer_chain_model.eval()
infer_chain_vocab = torch.load(args.infer_chain_vocab)

tokenizer = get_tokenizer('basic_english')
MIN_LEN = 5  # padding sequence if less than this

def get_infer_chain_feature(model, vocab, inp):
  features = []
  for i in range(inp.shape[0]):
    query = inp[i, 1].lower()
    pred_seq = inp[i, 2].lower().replace('.', ' ').replace('_', ' ')
    query = [vocab[token] for token in tokenizer(query)]
    pred_seq = [vocab[token] for token in tokenizer(pred_seq)]
    query = torch.tensor(query + [0] * (MIN_LEN - len(query))).unsqueeze(0)
    pred_seq = torch.tensor(pred_seq + [0] * (MIN_LEN - len(pred_seq))).unsqueeze(0)
    out1, out2 = model(query, pred_seq)
    d = F.pairwise_distance(out1, out2, p=2.0)
    features.append(d.item())
    if i == 0 or i % 5000 == 0:
      print('# Extracted inferential chain features: %d' % i)
  features = np.array(features)[:, np.newaxis]
  return features

infer_chain_data = np.loadtxt(args.infer_chain_data, delimiter='\t', dtype=str)
features = get_infer_chain_feature(infer_chain_model, infer_chain_vocab, infer_chain_data)
np.savetxt(args.type + '_features.txt', features, delimiter='\t')

