#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from urllib.parse import unquote
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

def split_train_test(dataset, label, test_size, train_size):
  indices = dataset.get_indices(label)
  np.random.shuffle(indices)
  test = torch.utils.data.Subset(dataset, indices[:test_size])
  train = torch.utils.data.Subset(dataset, indices[test_size:test_size + train_size])
  return train, test

# Original data files are downloaded from https://github.com/scottyih/STAGG
# For training core inferential chain model, only use the following two files:
#
# webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv
# webquestions.examples.train.e2e.top10.filter.q_ep.sid.tsv
#
# Columns of these two files are the same:
# F1 socre, pattern(question), predicate sequence(entity name & predicate sequence), question ID.
# See them in ./data.

RELATION_MATCHING_COLUMNS = ['f1_score', 'pattern', 'predicate_sequence', 'question_id']

class InferChainDataset(Dataset):
  def __init__(self, relation_matching_file):
    df = pd.read_csv(relation_matching_file, sep='\t', names=RELATION_MATCHING_COLUMNS)
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

# For simplicity, only use three features as input to the ranker: PatChain
# score, QuesEP score and entity linking score. The former two scores are
# predicted by relation matching models and saved in the matching scores file:
# the entity linking score is read from the entity linking file:
#
# matching_scores.txt
# webquestions.examples.train.e2e.top10.filter.sid.tsv
#
# Ground-truth labels (relevance) of parallel question and predicate sequence
# pairs are dependent on F1 score: if F1 > 0, label is 1 (relevant); if F1 == 0,
# label if 0 (irrelevant). F1 score are read from the relation matching file:
#
# webquestions.examples.train.e2e.top10.filter.q_ep.sid.tsv

ENTITY_LINKING_COLUMNS = ['question_id', 'mention', 'start_pos', 'length',
                          'linked_entity_id', 'entity_name', 'linking_score']
MATCHING_SCORE_COLUMNS = ['patchain_score', 'qep_score']

class RankingDataset(Dataset):
  def __init__(self, entity_linking_file,
               relation_matching_file, matching_score_file):
    df_e2e = pd.read_csv(entity_linking_file, sep='\t',
                         names=ENTITY_LINKING_COLUMNS)
    df_qep = pd.read_csv(relation_matching_file, sep='\t',
                         names=RELATION_MATCHING_COLUMNS)
    df_mat = pd.read_csv(matching_score_file, sep='\t',
                         names=MATCHING_SCORE_COLUMNS)

    df_e2e['entity_name'] = df_e2e['entity_name'].apply(
        self.__transform_entity_name)
    df_qep['predicate_sequence'] = df_qep['predicate_sequence'].apply(
        self.__transform_predicate_sequence)

    df_merged = pd.concat([df_qep, df_mat], axis=1)
    df_merged = df_merged.merge(
        df_e2e,
        left_on=['predicate_sequence', 'question_id'],
        right_on=['entity_name', 'question_id'])  # inner join two dataframes
    self.qid = df_merged['question_id'].unique()
    self.data = df_merged[['question_id', 'patchain_score', 'qep_score', 'linking_score']]
    self.target = (df_merged['f1_score'] > 0.).astype(int)

  def __transform_entity_name(self, name):
    return unquote(name).replace('_', ' ').lower()

  def __transform_predicate_sequence(self, pred):
    return re.sub(r'\s+(\w+\.)+\w+', '', pred).lower()

  def __len__(self):
    return len(self.qid)

  def __getitem__(self, idx):
    feature = self.data[self.data['question_id'] == self.qid[idx]].to_numpy()[:, 1:].astype(float)
    label = self.target[self.data['question_id'] == self.qid[idx]].to_numpy().astype(float)
    return feature, label

# data = RankingDataset('data/entity_linking_test.txt', 'data/qep_test.txt', 'data/matching_score_test.txt')
# data = RankingDataset('data/webquestions.examples.train.e2e.top10.filter.sid.tsv',
#                       'data/webquestions.examples.train.e2e.top10.filter.q_ep.sid.tsv',
#                       'data/matching_scores.txt')
# print(data[0])
# print(data[1])

