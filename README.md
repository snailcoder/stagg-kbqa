# stagg-kbqa

This repository reimplements some key components of staged query graph framework for KBQA, which is proposed in [Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base](https://aclanthology.org/P15-1128.pdf).
The original paper uses FreeBase and doesn't open source code. For simplicity, this repository only implements core inferential chain and ranking models based on the data provided [here](https://github.com/scottyih/stagg), and doesn't involve any knowledge graph operation.

## Core Inferential Chain
The goal of idenfitying core inferential chain is to map the natural utterance of the question to the correct predicate sequence. As the original paper does, I use a SiameseCNN for this purpose. Train the inferential chain model to calculate the PatChain score:
```bash
python3 train_infer_chain.py patchain ./data/webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv /path/to/save/model
```
Train the inferential chain model to calculate the QuesEP score:
```bash
python3 train_infer_chain.py patchain ./data/webquestions.examples.train.e2e.top10.filter.qep.sid.tsv /path/to/save/model
```
Now you've got PatChain and QuesEP models. If you want to use pretrained word embedding such as GLoVe and word2vec, please set --word_vec with your word embdding file.

## Feature Extraction
Before training the ranker, you need to extract features of PatChain and QuesEP. For example, if you want to extract PatChain features from ./data/webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv:
```bash
python3 extract_features.py patchain /path/of/patchain/model /path/of/patchain/vocab ./data/webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv
```
You'll get a patchain_features.txt file containing PatChain features in current directory.
Entity linking features can be loaded from ./data/webquestions.examples.train.e2e.top10.filter.sid.tsv directly. Other features such as ClueWeb, ConstraintEntityWord, ConstraintEntityInQ depending on the structure of knowledge graph are not involved in this repository.

## Learning to Rank
Given a query graph, once features of entity linking, PatChain and QuesEP are extracted, we can use them to train a LambdaRank model to assign a score for the query graph. Train the rank model:
```bash
python3 train_ranking.py ./data/webquestions.examples.train.e2e.top10.filter.sid.tsv data/webquestions.examples.train.e2e.top10.filter.q_ep.sid.tsv data/matching_scores.txt /path/to/save/model
```

## Inference
Because of absence of knowledge graph, I can't provide a complete inference for KBQA. However, once the ranker model is trained, you've owned the core component of this KBQA method. Given a query, you can load the ranker model to get score for each retrieved document:
```python
import torch

model = torch.load('rank_best_model.pth')
model.eval()

x = torch.rand(2, 3)  # assume the query retrieves two documents
y_score = model(x)
print(y_score)
```

