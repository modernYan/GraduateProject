#!/usr/bin/env python
# encoding: utf-8
"""
加载数据，获得训练集和测试集
"""
import csv
from sklearn.utils import shuffle

TRAIN_FILE = "./ag_news_csv/train.csv"
TEST_FILE = "./ag_news_csv/test.csv"


def parse_file(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ".  " + row[2])
        return texts, labels


def shuffle_data(train_values, labels):
    return shuffle(train_values, labels, random_state=0)


def load_data():
    (train_data, train_labels) = shuffle_data(*parse_file(TRAIN_FILE))
    valid_data = train_data[:10000]
    valid_labels = train_labels[:10000]
    train_data = train_data[10000:]
    train_labels = train_labels[10000:]
    (test_data, test_labels) = shuffle_data(*parse_file(TEST_FILE))
    print("加载训练数据集完毕,大小为:", len(train_data))
    print("加载开发数据集完毕,大小为:", len(valid_data))
    print("加载测试数据集完毕,大小为:", len(test_data))
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels
