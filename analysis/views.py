from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputForm
from django.core.exceptions import ValidationError
from django import forms
from .models import *
import json

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

def predict(net, test_review, file_Vocab, sequence_length=50):
    net.eval()

    test_ints = tokenize_review(test_review, file_Vocab)

    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)

    feature_tensor = torch.from_numpy(features)

    scores = net(feature_tensor)
    _, predictions = scores.max(1)

    return scores.cpu().detach().numpy()[0], predictions

def modelEmotion():
    a_file = open('analysis/emotion.pkl', "rb")
    vocab_to_int = pickle.load(a_file)

    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
    output_size = 13
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = RNN_LSTM(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)

    net.load_state_dict(torch.load('analysis/emotion_netFinal.pt', map_location=torch.device('cpu')))

    return net

def modelPosNeg():
    a_file = open('analysis/pos_neg.pkl', "rb")
    vocab_to_int = pickle.load(a_file)

    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
    output_size = 2
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = RNN_LSTM(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)

    net.load_state_dict(torch.load('analysis/pos_neg_netFinal.pt', map_location=torch.device('cpu')))

    return net

def home(request):
    return render(request,"home.html")

def getText(request):
    form = InputForm()
    return render(request,"index.html",context={'form': form })

def getAnalysis(request):
    form = InputForm()

    if request.method == "POST" :
        result = request.POST.get("submittedText")
        if result.isnumeric():
            error_msg="Enter a text"

            return render(request,'index.html',{'error':error_msg, 'form': form})
        else:
            global sentimenttext

            sentimenttext=sentimentText(submittedText=result)
            sentimenttext.save()

    net_PosNeg = modelPosNeg()
    net_Emotion = modelEmotion()

    result = result.replace('\n', '. ')

    array_PosNeg, pred_PosNeg = predict(net_PosNeg, result, 'analysis/pos_neg.pkl')
    array_Emotion, pred_Emotion = predict(net_Emotion, result, 'analysis/emotion.pkl')

    # print(array_Emotion)

    labelEmotion = ['Empty', 'Sadness', 'Enthusiasm', 'Neutral', 'Worry', 'Surprise', 'Love', 'Fun', 'Hate', 'Happiness', 'Boredom', 'Relief', 'Anger']
    labelSentiment = ["Positive", "Negative"]

    maxval_sentiment = array_PosNeg[pred_PosNeg]
    maxval_emotion = array_Emotion[pred_Emotion]

    array_PosNeg[array_PosNeg<0] = 0
    sum_PosNeg = np.sum(array_PosNeg)

    array_Emotion[array_Emotion<0] = 0
    sum_Emotion = np.sum(array_Emotion)

    dataPosNeg = []
    countA = 0
    for i in array_PosNeg:
        if i == maxval_sentiment:
            a = {
                'name': labelSentiment[countA],
                'y': ((i/sum_PosNeg)*100),
                'sliced': True,
                'selected': True
            }
        else:
            a = {
                'name': labelSentiment[countA],
                'y': ((i/sum_PosNeg)*100),
            }
        dataPosNeg.append(a)
        countA += 1

    dataEmot = []
    countB = 0
    for i in array_Emotion:
        if i == maxval_emotion:
            a = {
                'name': labelEmotion[countB],
                'y': ((i/sum_Emotion)*100),
                'sliced': True,
                'selected': True
            }
        else:
            a = {
                'name': labelEmotion[countB],
                'y': ((i/sum_Emotion)*100),
            }
        dataEmot.append(a)
        countB += 1

    dataReveiw = json.dumps(dataPosNeg)
    dataEmotion = json.dumps(dataEmot)

    sentimentFeedback = labelSentiment[pred_PosNeg]
    textEmotion = labelEmotion[pred_Emotion]


    dataContext = {
                'sentimentFeedback': sentimentFeedback,
                'textEmotion' : textEmotion,
                'dataReveiw' : dataReveiw,
                'dataEmotion' : dataEmotion,
                'result': result
                }
    return render(request, 'result.html', dataContext)

def feedback(request):
    if request.method=="POST":
        sentiment=request.POST.get("sentiment")
        emotions=request.POST.get("emotions")

        sr=sentimentReview(sentimentType=sentiment,emotionType=emotions,textReview=sentimenttext)
        sr.save()
        return render(request, 'thank.html')
