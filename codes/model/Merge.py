#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from codes.data import TEXT, train_iter, val_iter, TITLE


class MergeClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, attention_type=None):
        super(MergeClassifier, self).__init__()
        self.attention_type = attention_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.main_word_embeddings = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.main_word_embeddings.weight.data.copy_(TEXT.vocab.vectors)
        self.main_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.parse_word_embeddings = nn.Embedding(len(TITLE.vocab), embedding_dim)
        self.parse_word_embeddings.weight.data.copy_(TITLE.vocab.vectors)
        self.parse_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        if self.attention_type == 'bi_liner':
            self.bi_liner_W = Variable(torch.randn(self.hidden_dim, self.hidden_dim), requires_grad=True)

        if self.attention_type == 'concatention':
            self.concatention_W1 = Variable(torch.randn(self.hidden_dim, self.hidden_dim), requires_grad=True)
            self.concatention_W2 = Variable(torch.randn(self.hidden_dim, self.hidden_dim), requires_grad=True)
            self.concatention_b = Variable(torch.randn(self.hidden_dim, 1), requires_grad=True)
            self.concatention_v = Variable(torch.randn(1, self.hidden_dim), requires_grad=True)
            self.relu = nn.ReLU()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, main_batch, main_lengths, parse_batch, parse_lengths, sub_nums, parse_length_arg_sort):
        main_h0_c0 = self.init_hidden(main_batch.size(-1))
        main_embeds = self.main_word_embeddings(main_batch)
        main_lengths = main_lengths.view(-1).tolist()
        main_packed_input = pack_padded_sequence(main_embeds, main_lengths)
        main_outputs, (main_ht, main_ct) = self.main_lstm(main_packed_input, main_h0_c0)
        main_output = self.dropout_layer(main_ht[-1])

        parse_h0_c0 = self.init_hidden(parse_batch.size(-1))
        parse_embeds = self.parse_word_embeddings(parse_batch)
        parse_packed_input = pack_padded_sequence(parse_embeds, parse_lengths)
        parse_outputs, (parse_ht, parse_ct) = self.main_lstm(parse_packed_input, parse_h0_c0)
        raw_parse_output = self.dropout_layer(parse_ht[-1])
        parse_output = Variable(torch.zeros(raw_parse_output.data.shape))
        for index, sort_arg in enumerate(parse_length_arg_sort):
            parse_output[index] = raw_parse_output[sort_arg]
        start_index = 0
        for index, sub_num in enumerate(sub_nums):
            end_index = start_index + sub_num
            one_sentence = raw_parse_output[start_index:end_index]
            if index == 0:
                parse_output = self.attention(main_output[index], one_sentence)
            else:
                torch.cat((parse_output, self.attention(main_output[index], one_sentence)), dim=0)
            start_index = end_index

        output = main_output + parse_output
        output = self.hidden2out(output)
        output = self.log_softmax(output)
        return output

    def attention(self, u, one_sentence):
        u = u.view(1, -1)
        if not self.attention_type:
            return torch.mean(one_sentence, dim=0)
        one_sentence = one_sentence.t()
        if self.attention_type == 'dot':
            """
            score = u*m
            """
            score = torch.matmul(u, one_sentence)
        elif self.attention_type == 'bi_liner':
            """
            score = u*W*W_t*m
            """
            u_W = torch.matmul(u, self.bi_liner_W)
            u_W_W_T = torch.matmul(u_W, W.t())
            score = torch.matmul(u_W_W_T, one_sentence)
        elif self.attention_type == 'concatention':
            """
            score = V*activation(W1m+W2u+b)
            """
            relu_inner = torch.matmul(self.concatention_W1, one_sentence) + \
                         torch.matmul(self.concatention_W2, u.t()) + \
                         self.concatention_b
            score = torch.matmul(self.concatention_v, self.relu(relu_inner))

        p = self.softmax(score)
        result = torch.sum(torch.mul(p.t(), one_sentence.t()), dim=0, keepdim=True)
        return result
