!pip install -q pytorch-lightning
!pip install -q deep-translator
!pip install -q youtube_transcript_api
!pip install -q streamlit
!pip install -q plotly
!python -m spacy download en_core_web_md -q

import datetime
import glob
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from deep_translator import GoogleTranslator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from youtube_transcript_api import YouTubeTranscriptApi

class ModelWithoutMRC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.padding = (3, 0)
        self.stride = (6, 1)
        self.kernel = (6, 1)
        self.GRU = nn.GRU(300, 300, batch_first=True)
        self.GRU1 = nn.GRU(300, 300, batch_first=True)
        self.fc1 = nn.Linear(200 * 300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)


    def forward(self, vectors, mrc):
        x = vectors.unsqueeze(1)
        x = self.GRU(x.squeeze(1))[0]
        x = self.GRU1(x)[0]
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


    def training_step(self, batch, batch_idx):
        inputs, mrc, labels = batch
        outputs = self(inputs, mrc)

        loss = self.loss(outputs, labels.float())

        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, mrc, labels = batch
        outputs = self(inputs, mrc)

        loss = self.loss(outputs, labels.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        inputs, mrc, labels = batch
        outputs = F.sigmoid(self(inputs, mrc))

        mse = torch.mean((outputs - labels)**2)
        self.log('test_mse', mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)


class ModelWithMRC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.padding = (3, 0)
        self.stride = (6, 1)
        self.kernel = (6, 1)
        self.GRU = nn.GRU(300, 300, batch_first=True)
        self.GRU1 = nn.GRU(300, 300, batch_first=True)
        self.fc1 = nn.Linear(200 * 300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128 + 27, 64)
        self.fc4 = nn.Linear(64, 1)


    def forward(self, vectors, mrc):
        x = vectors.unsqueeze(1)
        x = self.GRU(x.squeeze(1))[0]
        x = self.GRU1(x)[0]
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, mrc], dim=1)))
        x = self.fc4(x)

        return x


    def training_step(self, batch, batch_idx):
        inputs, mrc, labels = batch
        outputs = self(inputs, mrc)

        loss = self.loss(outputs, labels.float())

        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, mrc, labels = batch
        outputs = self(inputs, mrc)

        loss = self.loss(outputs, labels.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        inputs, mrc, labels = batch
        outputs = F.sigmoid(self(inputs, mrc))

        mse = torch.mean((outputs - labels)**2)
        self.log('test_mse', mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)

class FinalModel(pl.LightningModule):
    def __init__(self, models_info):
        super().__init__()
        
        self.loss = torch.nn.BCEWithLogitsLoss()
        for model_name in ["e", "n", "a", "c", "o"]:
            model_info = models_info[model_name]
            if model_info["with_mrc"]:
                model = ModelWithMRC()
            else:
                model = ModelWithoutMRC()
            model.load_state_dict(model_info["state_dict"])
            setattr(self, model_name, model)

    def forward(self, v, m):
        return torch.cat([self.e(v, m), self.n(v, m), self.a(v, m), self.c(v, m), self.o(v, m)], dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

indices_to_labels = { 0:'e', 1:'n', 2:'a', 3:'c', 4:'o' }
labels_to_indices = { 'e':0, 'n':1, 'a':2, 'c':3, 'o':4 }

models_info = torch.load("./models_info.pth")
fModel = FinalModel(models_info)

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

with open(f'./mrc2_dict.json', 'r') as f:
    word_data = json.load(f)
    
def evalText(text):
  fModel.eval()
  fModel.to(device)
  with torch.no_grad():
      input = torch.zeros((200, 300), device=device, dtype=torch.float32)
      mrc = np.zeros(27)
      words = []
      for word in nlp(text)[:200]:
        if word.lemma_ in word_data:
          mrc += np.array(word_data[word.lemma_])
        if word.has_vector and word.is_alpha:
          words.append(word.vector)

      mrc = torch.tensor(np.array(mrc), device=device, dtype=torch.float32).unsqueeze(0)
      mrc = mrc / torch.norm(mrc, p=2, dim=1, keepdim=True)
      mrc = torch.nan_to_num(mrc, nan=0)

      words = torch.tensor(np.array(words), device=device, dtype=torch.float32)
      input[:words.shape[0]] = words
      input = input.unsqueeze(0)

      outputs = F.sigmoid(fModel(input, mrc)).squeeze()
      return outputs

def drawplt1(data):
  sns.set_style("whitegrid")

  plt.figure(figsize=(8, 6))
  plt.xlim(0, 100)
  sns.barplot(x=data, y=label_names, hue=label_names, legend=False)

  for i, v in enumerate(data):
      vv = round(v, 2)
      plt.text(vv + 1, i, str(vv)+'%', color='black', va='center', fontsize=12)

  plt.show()

def drawplt2(data):
  ln = label_names.copy()
  ln.append(ln[0])
  lab = (np.array(raw_predicted_labels) * 100).tolist()
  lab.append(lab[0])

  df = pd.DataFrame(dict(r=lab, theta=ln))
  fig = px.line_polar(df, r='r', theta='theta', line_close=True)
  fig.update_traces(fill='toself')
  fig.update_layout(
      polar=dict(
          radialaxis=dict(
              tickvals=[0, 20, 40, 60, 80, 100],
              ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
              range=[0, 100]
          )
      )
  )
  fig.show()

def big5_to_mbti(big5):
    res = ""
    res += "e" if big5[0] >= 0.5 else "i"
    res += "n" if big5[4] >= 0.5 else "s"
    res += "f" if big5[2] >= 0.5 else "t"
    res += "j" if big5[3] >= 0.5 else "p"
    return res.upper()

label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
text = "Hi. What is your name? How old are you? Where are u from Hello fella brother"
text = "dont have nothing for you"
text = GoogleTranslator(source='auto', target='en').translate(text)

outputs = evalText(text)
raw_predicted_labels = outputs.float().tolist()
predicted_labels = (outputs > 0.5).float().tolist()
result1 = {label_names[i]: predicted_labels[i] for i in range(len(label_names))}
result2 = {label_names[i]: raw_predicted_labels[i] for i in range(len(label_names))}
print(big5_to_mbti(raw_predicted_labels))
drawplt1((outputs.float() * 100).tolist())
drawplt2((outputs.float() * 100).tolist())
print(result1)
print(result2)
print()