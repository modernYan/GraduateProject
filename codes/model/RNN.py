#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.data import TEXT


class RNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(RNNClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.word_embeddings.weight.data.copy_(TEXT.vocab.vectors)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=1)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        return autograd.Variable(torch.randn(1, batch_size, self.hidden_dim))

    def forward(self, batch, lengths):
        h0 = self.init_hidden(batch.size(-1))
        embeds = self.word_embeddings(batch)
        lengths = lengths.view(-1).tolist()
        packed_input = pack_padded_sequence(embeds, lengths)
        outputs, ht = self.rnn(packed_input, h0)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)
        return output
