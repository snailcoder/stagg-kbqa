#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_patchain.py
# Author            : Yan <yanwong@126.com>
# Date              : 09.07.2021
# Last Modified Date: 29.04.2022
# Last Modified By  : Yan <yanwong@126.com>

'''
Train a siamese networks for computing PatChain feature which compares the
pattern (replacing the topic entity with an entity symbol) and the predicate
sequence. The original peper seems to compute a binary classification score,
unlike that, I compute the distance of pattern and predicate sequence directly.
'''
import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import models
import datasets
import config
import utils

parser = argparse.ArgumentParser(
    description='Train Siamese CNN model for computing similarity between '
                'question pattern and predicate sequence, save the best weights.')
parser.add_argument('data_file', help='dataset file')
parser.add_argument('save_dir', help='directory to save vocab and model')
parser.add_argument('--word_vec', help='pretrained word vector file')
parser.add_argument('--log_interval', type=int, default=100,
                    help='print training log every interval batches')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

dataset = datasets.InferChainDataset(args.data_file)

model_config = config.ModelConfig()
train_config = config.TrainConfig()

train_pos, test_pos = datasets.split_train_test(dataset, 0,
    train_config.test_size_per_class, train_config.train_size_per_class)
train_neg, test_neg = datasets.split_train_test(dataset, 1,
    train_config.test_size_per_class, train_config.train_size_per_class)

train_data = torch.utils.data.ConcatDataset([train_pos, train_neg])
test_data = torch.utils.data.ConcatDataset([test_pos, test_neg])

tokenizer = get_tokenizer('basic_english')
vocab = utils.build_vocab(train_data, tokenizer)
print('Vocab size: ', len(vocab))

def save_model(model, save_dir, filename):
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, filename)
  torch.save(model.state_dict(), save_path)
  torch.save(model, save_path)

save_model(vocab, args.save_dir, 'vocab.pth')

text_transform = lambda x: [vocab[token] for token in tokenizer(x)]

def collate_batch(batch):
  pattern_batch, predicate_sequence_batch, label_batch = [], [], []
  for (pat, pred_seq, label) in batch:
    pattern_batch.append(torch.tensor(text_transform(pat)))
    predicate_sequence_batch.append(torch.tensor(text_transform(pred_seq)))
    label_batch.append(label)
  pattern_batch = pad_sequence(pattern_batch, batch_first=True)
  predicate_sequence_batch = pad_sequence(predicate_sequence_batch, batch_first=True)
  label_batch = torch.tensor(label_batch)
  return pattern_batch, predicate_sequence_batch, label_batch

train_dataloader = DataLoader(train_data,
                              batch_size=train_config.batch_size,
                              shuffle=train_config.shuffle,
                              collate_fn=collate_batch)
test_dataloader = DataLoader(test_data,
                             batch_size=train_config.batch_size,
                             shuffle=train_config.shuffle,
                             collate_fn=collate_batch)


pretrained_embedding = None
if args.word_vec:
  pretrained_embedding = utils.load_pretrained_embedding(
      args.word_vec, model_config.d_word, vocab.get_itos())
  print('# total words: %d' % len(pretrained_embedding))
  pretrained_embedding = torch.FloatTensor(pretrained_embedding).to(device)

model = models.SiameseCnn(model_config, len(vocab), pretrained_embedding)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (x1, x2, y) in enumerate(dataloader):
    optimizer.zero_grad()
    out1, out2 = model(x1, x2)
    loss = loss_fn(out1, out2, y)
    loss.backward()
    optimizer.step()

    if batch > 0 and batch % 10 == 0:
      loss, current = loss.item(), batch * len(y)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for x1, x2, y in dataloader:
      out1, out2 = model(x1, x2)
      test_loss += loss_fn(out1, out2, y).item()
      d = F.pairwise_distance(out1, out2, p=2.0)
      correct += torch.eq(d >= 1.5, y == 1).type(torch.float).sum().item()
      
  test_loss /= num_batches
  accuracy = correct / size
  print(f'Test Error: \n Accuracy: {(100*accuracy):>0.1f}%,'
        f' Avg loss: {test_loss:>7f} \n')
  return accuracy


loss_fn = models.ContrastiveLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=train_config.learning_rate)

best_accu = 0
for t in range(train_config.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    accu = test_loop(test_dataloader, model, loss_fn)
    if accu > best_accu:
      best_accu = accu
      save_model(model, args.save_dir, 'best_model.pth')
      print(f'Best accuracy: {(100*best_accu):>0.1f}%\n')
print(f'Global best accuracy: {(100*best_accu):>0.1f}%')

