#!/usr/bin/env python
# encoding: utf-8
from pprint import pprint

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from codes.config import EMBEDDING_DIM, HIDDEN_SIZE, LABEL_SIZE, LEARNING_RATE, EPOCH
from codes.data import train_iter, val_iter, test_iter, TITLE
from codes.model.LSTM import LSTMClassifier
from codes.model.Merge import MergeClassifier
from codes.model.RNN import RNNClassifier
from codes.parse.parser import ParseSentence


class TrainModel:
    def __init__(self, model):
        self.model = model
        self.max_dev_accuracy = 0.0
        self.saved_model_path = './saved_model/{}.pkl'.format(model.__class__.__name__)
        self.calculate_dev_step = 20
        self.parsentence = ParseSentence(has_parsed=True)
        print('model init successfully\nThe model is the following:')
        pprint(model)

    def follow(self, data):
        (x, x_lengths), y = data.text, data.label
        if isinstance(model, MergeClassifier):
            parse_batch, parse_lengths, sub_nums, parse_length_arg_sort = self.get_parse_result(data.title)
            y_pre = self.model(x, x_lengths, parse_batch, parse_lengths, sub_nums, parse_length_arg_sort)
        else:
            y_pre = self.model(x, x_lengths)
        return y_pre

    def train(self):
        print('START TRAINING...')
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCH):
            step_count = 1
            for data in tqdm(train_iter):
                self.model.zero_grad()
                y_pre = self.follow(data)
                loss = loss_function(y_pre, data.label - 1)
                loss.backward()
                optimizer.step()
                if step_count % self.calculate_dev_step == 0:
                    dev_batch = next(iter(val_iter))
                    current_dev_accuracy = self.calculate_accuracy(dev_batch)
                    self.print_log(epoch, step_count, loss.data[0], current_dev_accuracy)
                    self.try_save_model(current_dev_accuracy)
                step_count += 1

    def get_parse_result(self, title):
        title_length = title[1]
        title = title[0]
        title.t_()
        title = title.data.tolist()
        sub_nums = []
        raw_parse_batch = []
        raw_parse_lengths = []
        for sentence, sentence_length in zip(title, title_length):
            sub_sentences = self.parsentence.sentence_dic[tuple(sentence[:sentence_length])]
            assert len(sub_sentences) > 0
            sub_nums.append(len(sub_sentences))
            for sub_sentence in sub_sentences:
                sub_sentence_list = sub_sentence.split()
                raw_parse_batch.append([TITLE.vocab.stoi[word] for word in sub_sentence_list])
                raw_parse_lengths.append(len(sub_sentence_list))
        parse_length_arg_sort = list(np.argsort(raw_parse_lengths))
        parse_length_arg_sort.reverse()
        parse_batch = [raw_parse_batch[index] for index in parse_length_arg_sort]
        parse_lengths = [raw_parse_lengths[index] for index in parse_length_arg_sort]
        parse_batch_array = np.ones((len(parse_batch), max(parse_lengths)), dtype=np.int32)
        for i in range(len(parse_batch)):
            parse_batch_array[i, 0:len(parse_batch[i])] = parse_batch[i]
        parse_batch_variable = torch.autograd.Variable(torch.LongTensor(parse_batch_array))
        parse_batch_variable.t_()
        return parse_batch_variable, parse_lengths, sub_nums, parse_length_arg_sort

    def print_log(self, epoch, step, loss_value, dev_accuracy):
        print('\nepoch:{}\tstep:{}\ttrain_loss:{:.4f}\tdev_accuracy:{:.4f}\t\tdev_max_accuracy:{:.4f}'.format(
            epoch, step, loss_value, dev_accuracy, self.max_dev_accuracy
        ))

    def test(self):
        print('START TESTING...')
        self.model.load_state_dict(torch.load(self.saved_model_path))
        test_batch = next(iter(test_iter))
        accuracy = self.calculate_accuracy(test_batch)
        print('test accuracy {}'.format(accuracy))

    def calculate_accuracy(self, batch):
        dev_y_pre = self.follow(batch)
        dev_y_pre = torch.max(dev_y_pre, 1)[1].data.numpy().squeeze()
        accuracy = sum(dev_y_pre == batch.label.data - 1) / float(batch.label.shape[0])
        return accuracy

    def try_save_model(self, current_dev_accuracy):
        if current_dev_accuracy > self.max_dev_accuracy:
            self.max_dev_accuracy = current_dev_accuracy
            print('save model,current dev accuracy {}'.format(current_dev_accuracy))
            torch.save(self.model.state_dict(), self.saved_model_path)


if __name__ == "__main__":
    model = MergeClassifier(EMBEDDING_DIM, HIDDEN_SIZE, LABEL_SIZE, 'concatention')
    trainModel = TrainModel(model)
    trainModel.train()
