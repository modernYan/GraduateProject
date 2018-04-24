#!/usr/bin/env python
# encoding: utf-8
import torch
from nltk import CoreNLPDependencyParser, Tree, pprint
import pickle

from tqdm import tqdm

from codes.config import ROOT_PATH
from codes.data import train_iter, glove, TITLE, val_iter, test_iter, TEXT


class ParseSentence:
    def __init__(self, has_parsed=False):
        self.has_parsed = has_parsed
        if has_parsed:
            with open(ROOT_PATH + '/parse/parse.pkl', 'rb') as f:
                self.sentence_dic = pickle.load(f)
        else:
            self.sentence_dic = {}

    def parse_sub_structure(self, sentence):
        if self.has_parsed:
            return self.sentence_dic[sentence]
        else:
            # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
            dep_parser = CoreNLPDependencyParser(url='http://127.0.0.1:9000')
            parser, = dep_parser.raw_parse(sentence)
            all_path = []

            def walk(path, tree):
                for node in tree:
                    if type(node) is Tree:
                        path += (' ' + node.label())
                        walk(path, node)
                    else:
                        path += (' ' + node)
                        all_path.append(path)

            tree = parser.tree()
            path = tree.label()
            walk(path, tree)
            # pprint(all_path)
            # tree.draw()
            return all_path

    def parse_all_sentence_and_save(self):
        for batch in tqdm(train_iter):
            self._parse_one_batch(batch)
        for batch in tqdm(test_iter):
            self._parse_one_batch(batch)
        for batch in tqdm(val_iter):
            self._parse_one_batch(batch)
        with open(ROOT_PATH + '/parse/parse.pkl', 'wb') as f:
            pickle.dump(self.sentence_dic, f)

    def _parse_one_batch(self, batch):
        (x, x_length), (text, text_length) = batch.title, batch.text
        x.t_()
        x = x.data.tolist()
        text = text.t_().data.tolist()
        x_length = x_length.tolist()
        for sentence, sentence_length, text in zip(x, x_length, text):
            sentence_str = " ".join(TITLE.vocab.itos[word_index] for word_index in sentence[:sentence_length])
            sub_sentence = self.parse_sub_structure(sentence_str)
            if len(sub_sentence) == 0:
                sub_sentence = [sentence_str]
            self.sentence_dic[tuple(sentence[:sentence_length])] = sub_sentence


if __name__ == "__main__":
    parsentence = ParseSentence()
    parsentence.parse_all_sentence_and_save()
