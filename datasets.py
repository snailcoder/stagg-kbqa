#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datasets.py
# Author            : Yan <yanwong@126.com>
# Date              : 23.04.2022
# Last Modified Date: 29.04.2022
# Last Modified By  : Yan <yanwong@126.com>

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

# Original data files are downloaded from https://github.com/scottyih/STAGG
# For training core inferential chain model, only use the following two files:
#
# webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv
# webquestions.examples.train.e2e.top10.filter.q_ep.sid.tsv
#
# Columns of these two files are the same:
# F1 socre, pattern(question), predicate sequence(entity name & predicate sequence), question ID.
# See them in ./data.

COLUMN_NAMES = ['f1_score', 'pattern', 'predicate_sequence', 'question_id']

class InferChainDataset(Dataset):
  def __init__(self, annotations_file):
    df = pd.read_csv(annotations_file, sep='\t', names=COLUMN_NAMES)
    # The paper said only use chain-only query graphs that achieves F1=0.5
    # to form the parallel question and predicate sequence pairs. However, for
    # training siamese networks, we need not only positive pairs but also
    # negative pairs. In fact, negative pairs are those completely incorrect
    # query graphs of which F1 scores are 0. So here retains pairs with F1=0
    # and F1>=0.5.
    df = df[(df['f1_score'] >= 0.5) | (df['f1_score']== 0.0)].reset_index(drop=True)
    self.data = df.loc[:, ['pattern', 'predicate_sequence']]
    # Positive pairs have label 0, negative pairs have label 1.
    self.targets = (df['f1_score'] == 0.0).astype(int)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    pat = self.data.iloc[idx].pattern.lower()
    pred_seq = self.data.iloc[idx].predicate_sequence.lower()
    pred_seq = pred_seq.replace('.', ' ').replace('_', ' ')
    label = self.targets.iloc[idx]
    return pat, pred_seq, label

  def get_indices(self, label):
    return np.array(self.targets[self.targets == label].index)

def split_train_test(dataset, label, test_size, train_size):
  indices = dataset.get_indices(label)
  np.random.shuffle(indices)
  test = torch.utils.data.Subset(dataset, indices[:test_size])
  train = torch.utils.data.Subset(dataset, indices[test_size:test_size + train_size])
  return train, test

