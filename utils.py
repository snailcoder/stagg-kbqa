#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : utils.py
# Author            : Yan <yanwong@126.com>
# Date              : 28.04.2022
# Last Modified Date: 29.04.2022
# Last Modified By  : Yan <yanwong@126.com>

from collections import Counter, OrderedDict

import numpy as np

from torchtext.vocab import Vocab, vocab

def build_vocab(dataset, tokenizer):
  counter = Counter()
  for x1, x2, _ in dataset:
    counter.update(tokenizer(x1))
    counter.update(tokenizer(x2))

  sorted_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
  ordered_dict = OrderedDict(sorted_tuples)
  special_tokens = ['<PAD>', '<UNK>']
  v = vocab(ordered_dict, specials=special_tokens, special_first=True)
  v.set_default_index(v['<UNK>'])
  return v

def load_pretrained_embedding(filename, emb_dim, words):
  w2v = {}
  ws = set(words)
  with open(filename, 'r', encoding='utf-8') as f:
    for row in f:
      toks = row.split(' ')
      if toks[0] in ws:
        w2v[toks[0]] = np.array(list(map(float, toks[-emb_dim:])))
  emb = []
  oop_num = 0
  for w in words:
    if w not in w2v:
      w2v[w] = np.random.uniform(-0.25, 0.25, emb_dim)
      oop_num += 1
    emb.append(w2v[w])

  print(f'# out-of-pretrained words: {oop_num}')

  return emb

