#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
  def __init__(self, margin=2.0):
    super(ContrastiveLoss, self).__init__()
    
    self.margin = margin

  def forward(self, x1, x2, y):
    # N: batch size
    # D: vector dimension

    # x1.shape == x2.shape == (N, D)
    # y.shape == (N)

    d = F.pairwise_distance(x1, x2, p=2.0)  # (N)

    # When training, if x1 and x2 are of the same class, y should be 0;
    # if x1 and x2 are of different class, y should be 1.
    # loss = (1 - y) * d ^ 2 + y * max(margin - d, 0) ^ 2

    loss = (1 - y) * torch.pow(d, 2) + y * torch.pow(
        torch.clamp(self.margin - d, min=0.0), 2)
    loss = torch.mean(loss)
    return loss

def dcg_score(y_true, y_score, k=None):
  # y_true: ground-truth labels (grade) of retrieved documents to queries
  # y_score: predict scores of queries
  # y_true.shape == y_score.shape == (n_query, n_doc)

  discount = 1 / torch.log2(torch.arange(y_true.size()[1]) + 2)
  if k is not None:
    discount[k:] = 0
  ranking = torch.argsort(y_score, descending=True)
  ranked = y_true[torch.arange(ranking.size()[0]).unsqueeze(1), ranking]
  gain = torch.pow(2.0, ranked) - 1
  return torch.matmul(discount, gain.t())  # (n_query,)

def idcg_score(y_true, k=None):
  return dcg_score(y_true, y_true, None)

def ndcg_score(y_true, y_score, k=None):
  dcg = dcg_score(y_true, y_score, k)
  idcg = idcg_score(y_true, k)
  ndcg = torch.zeros_like(dcg)
  mask = idcg != 0
  ndcg[mask] = dcg[mask] / idcg[mask]
  return ndcg

def ranknet_loss(y_true, y_score):
  # y_true: ground-truth labels (grade) of retrieved documents to queries
  # y_score: predict scores of queries
  # y_true.shape == y_score.shape == (n_query, n_doc)

  c = torch.zeros((y_true.size()[0],))
  for i in range(y_true.size()[1] - 1):
    for j in range(i + 1, y_true.size()[1]):
      p = torch.zeros((y_true.size()[0],))
      p[y_true[:, i] > y_true[:, j]] = 1
      p[y_true[:, i] < y_true[:, j]] = -1
      p = (1 + p) / 2
      delta_score = y_score[:, i] - y_score[:, j]
      # Compute cost of each document pair (i, j), according to equation (3) of
      # https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
      # (Learning to Rank using Gradient Descent)
      # and equation (8) of
      # https://papers.nips.cc/paper/2006/file/af44c4c56f385c43f2529f9b1b018f6a-Paper.pdf
      # (Learning to Rank with Nonsmooth Cost Functions)
      # c_ij = -p * delta_score + torch.log2(1 + torch.exp(delta_score))
      # Use log-sum-trick to avoid overflow and torch values go to inf.
      delta_max = torch.zeros_like(delta_score)
      delta_max[delta_score > 0] = delta_score[delta_score > 0]
      c_ij = -p * delta_score + delta_max / 0.6931 + torch.log2(
          torch.exp(-delta_max) + torch.exp(delta_score - delta_max))
      # Don’t accumulate history across your training loop, or you may exhaust memory.
      # https://pytorch.org/docs/stable/notes/faq.html
      c += float(c_ij)  # use float() to fix 

  total = y_true.size()[0] * y_true.size()[1] * (y_true.size()[1] - 1) / 2
  return torch.sum(c) / total

# y_score = torch.tensor([[.1, .2, .3, 4, 70], [.1, .3, 70, 4, .2]])
# y_true = torch.tensor([[10,  0,  0,  1,  5], [ 1,  3,  5,  2,  4]])
# 
# dcg = dcg_score(y_true, y_score)
# print(dcg)
# idcg = idcg_score(y_true)
# print(idcg)
# rl = ranknet_loss(y_true, y_score)
# print(rl)

