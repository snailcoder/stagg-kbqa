#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class PatChainConfig(object):
  def __init__(self):
    # Model configs
    self.d_word = 300
    self.static_embedding = False
    self.filter_heights = [3, 4, 5]
    self.filter_num = [200, 200, 200]
    self.dropout = 0.5
    self.d_output = 2

    # Training configs
    self.learning_rate = 1e-3
    self.batch_size = 16
    self.epochs = 20
    self.shuffle = True
    self.train_size_per_class = 5000
    self.test_size_per_class = 300
    self.dist_of_same_class = 1.5

class QuesEPConfig(object):
  def __init__(self):
    # Model configs
    self.d_word = 300
    self.static_embedding = False
    self.filter_heights = [3, 4, 5]
    self.filter_num = [200, 200, 200]
    self.dropout = 0.5
    self.d_output = 2

    # Training configs
    self.learning_rate = 1e-3
    self.batch_size = 16
    self.epochs = 20
    self.shuffle = True
    self.train_size_per_class = 5000
    self.test_size_per_class = 300
    self.dist_of_same_class = 1.5

class RankerConfig:
  def __init__(self):
    self.d_input = 3
    self.d_hidden = 100
    self.k = None

    self.learning_rate = 1e-3
    self.epochs = 1
    self.train_size = 0.9

