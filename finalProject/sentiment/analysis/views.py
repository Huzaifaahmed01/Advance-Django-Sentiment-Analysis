from django.shortcuts import render
import numpy as np
import re
from string import punctuation
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

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
    a_file = open(fileVocab, "rb")
    vocab_to_int = pickle.load(a_file)

    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

def pad_features(reviews_ints, seq_length=50):

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

def predict(net, test_review, sequence_length=50):
    net.eval()

    test_ints = tokenize_review(test_review)

    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)

    feature_tensor = torch.from_numpy(features)

    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    scores = net(feature_tensor)
    _, predictions = scores.max(1)

    return scores.cpu().detach().numpy()[0], predictions

def modelEmotion():
    a_file = open('emotion.pkl', "rb")
    vocab_to_int = pickle.load(a_file)

    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
    output_size = 13
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = RNN_LSTM(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)

    net.load_state_dict(torch.load('emotion_netFinal.pt'))

    return net

def modelPosNeg():
    a_file = open('pos_neg.pkl', "rb")
    vocab_to_int = pickle.load(a_file)

    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
    output_size = 13
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = RNN_LSTM(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)

    net.load_state_dict(torch.load('pos_neg_netFinal.pt'))

    return net
# Create your views here.
