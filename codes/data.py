#!/usr/bin/env python
# encoding: utf-8
import os
from torchtext import vocab, data
import re

from codes.config import BATCH_SIZE, ROOT_PATH

glove = vocab.GloVe(name='6B', dim=100, cache=ROOT_PATH + '/.vector_cache')


def clean_str(string):
    string = " ".join(string)
    string = re.sub(r"[^A-Za-z0-9(),!?@]", " ", string)
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().split()


TEXT = data.Field(lower=True, sequential=True, include_lengths=True)
TITLE = data.Field(lower=True, sequential=True, include_lengths=True, preprocessing=clean_str)
LABELS = data.Field(sequential=False, use_vocab=False, batch_first=True)

train, val, test = data.TabularDataset.splits(
    path=ROOT_PATH + '/ag_news_csv/', train='train.csv',
    validation='dev.csv', test='test.csv', format='csv',
    fields=[('label', LABELS), ('title', TITLE), ('text', TEXT)])

TEXT.build_vocab(train, val, test, vectors=glove)
TITLE.build_vocab(train, val, test, vectors=glove)
LABELS.build_vocab(train, val, test)

train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_sizes=(BATCH_SIZE, len(val), len(test)),
    sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False
)
