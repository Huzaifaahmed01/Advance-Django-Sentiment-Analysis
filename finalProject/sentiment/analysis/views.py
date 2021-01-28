from django.shortcuts import render
import numpy as np
import re
from string import punctuation
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drop_prob)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=0.4, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)

        if (train_on_gpu):
          # Set initial hidden and cell states
          h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
          c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
          # Set initial hidden and cell states
          h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
          c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(embeds, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out[:,-1,:]
        out = self.dropout(out)
        # print(out[:,-1,:].shape)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

def tokenize_review(test_review, fileVocab):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

def pad_features(reviews_ints, seq_length):

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

# Create your views here.
