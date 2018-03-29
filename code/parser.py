#!/usr/bin/env python
# encoding: utf-8
import pickle
from nltk import CoreNLPDependencyParser, Tree, pprint
from nltk.parse.stanford import StanfordDependencyParser
import os

# os.environ['STANFORD_PARSER'] = './stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = './stanford-parser-3.9.1-models.jar'
# model_path = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
#
# dep_parser = StanfordDependencyParser(model_path=model_path)
from data import load_data
import pickle

with open('parse.pkl', 'rb') as f:
    data_d = pickle.load(f)


def parse_sub_structure(sentence):
    # dep_parser = CoreNLPDependencyParser(url='http://10.62.143.113:9000')
    # parser, = dep_parser.raw_parse(sentence)
    # all_path = []
    #
    # def walk(path, tree):
    #     for node in tree:
    #         if type(node) is Tree:
    #             path += (' ' + node.label())
    #             walk(path, node)
    #         else:
    #             path += (' ' + node)
    #             all_path.append(path)
    #
    # tree = parser.tree()
    # path = tree.label()
    # walk(path, tree)
    # # pprint(all_path)
    # # tree.draw()
    # return all_path
    return data_d[sentence]


if __name__ == "__main__":
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_data()
    train_data.extend(list(valid_data))
    train_data.extend(list(test_data))
    d = {}
    left_count = len(train_data)
    for each_sentence in train_data:
        print('left {}'.format(left_count))
        d[each_sentence] = parse_sub_structure(each_sentence)
        left_count -= 1
    with open('parse.pkl', 'wb') as f:
        pickle.dump(d, f)
