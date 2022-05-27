#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import nn
import metrics 

class LambdaRank(nn.Module):
  def __init__(self, config):
    super(LambdaRank, self).__init__()
    self.linear_layers = nn.Sequential(
        nn.Linear(config.d_input, config.d_hidden),
        nn.ReLU(),
        nn.Linear(config.d_hidden, 1))
    self.k = config.k

  def forward(self, x):
    # x.shape == (batch_size, d_input)

    logits = self.linear_layers(x)
    return logits

  def get_lambda_for_one_query(self, y_true, y_score):
    # For a given query, y_true are ground-truth labels (grade) of retrieved
    # documents, y_score are predict scores of those documents.
    # y_true.shape == (n_doc, 1)
    # y_score.shape == (n_doc, 1)

    n_doc = y_true.size()[0]
    y_true = torch.reshape(y_true, (n_doc,))
    y_score = torch.reshape(y_score, (n_doc,))

    lambdas = torch.zeros(n_doc)
    sorted_idx = torch.argsort(y_score, descending=True)
    ranking = torch.zeros(n_doc)
    ranking[sorted_idx] = 1 + torch.arange(n_doc, dtype=torch.float)
    max_dcg = metrics.idcg_score(y_true.view((1, n_doc)), self.k)
    inverse_max_dcg = torch.zeros_like(max_dcg)
    mask = max_dcg != 0
    inverse_max_dcg[mask] = 1 / max_dcg[mask]
    for i in range(n_doc - 1):
      for j in range(i + 1, n_doc):
        if y_true[i] == y_true[j]:
          continue
        if y_true[i] > y_true[j]:
          high, low = i, j
        else:
          high, low = j, i

        delta_discount = 1 / torch.log2(1 + ranking[high]) - 1 / torch.log2(1 + ranking[low])
        delta_gain = torch.pow(2., y_true[high]) - torch.pow(2., y_true[low])
        p = torch.sigmoid(-(y_score[high] - y_score[low]))
        lamb = inverse_max_dcg * p * delta_discount * delta_gain
        lamb = torch.abs(lamb)
        lambdas[high] = lambdas[high] + lamb
        lambdas[low] = lambdas[low] - lamb

    lambdas = torch.reshape(lambdas, (n_doc, 1))
    return lambdas

# config = Config()
# model = LambdaRank(config)
# y_true = torch.tensor([
#   [1.], [1.], [1.], [1.], [1.],
#   [0.], [0.], [0.], [0.], [0.],
#   [0.], [0.], [0.], [0.], [0.],
#   [0.], [0.], [0.], [0.], [0.]])
# y_score = torch.tensor([
#   [-0.1421], [-0.2051], [-0.1832], [-0.1757], [-0.1740],
#   [-0.1604], [-0.1434], [-0.1816], [-0.1456], [-0.1506],
#   [-0.1759], [-0.1394], [-0.1380], [-0.1398], [-0.1580],
#   [-0.1580], [-0.1524], [-0.1297], [-0.1601], [-0.1317]])
# lambs = model.get_lambda_for_one_query(y_true, y_score)
# print(lambs)

