import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import requests
from io import BytesIO

class ModelWithoutNRC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.GRU = nn.GRU(300, 300, batch_first=True)
        self.GRU1 = nn.GRU(300, 300, batch_first=True)
        self.fc1 = nn.Linear(200 * 300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)


    def forward(self, vectors, nrc):
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
        inputs, nrc, labels = batch
        outputs = self(inputs, nrc)

        loss = self.loss(outputs, labels.float())

        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, nrc, labels = batch
        outputs = self(inputs, nrc)

        loss = self.loss(outputs, labels.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        inputs, nrc, labels = batch
        outputs = F.sigmoid(self(inputs, nrc))

        mse = torch.mean((outputs - labels)**2)
        self.log('test_mse', mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)


class ModelWithNRC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.GRU = nn.GRU(300, 300, batch_first=True)
        self.GRU1 = nn.GRU(300, 300, batch_first=True)
        self.fc1 = nn.Linear(200 * 300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128 + 27, 64)
        self.fc4 = nn.Linear(64, 1)


    def forward(self, vectors, nrc):
        x = vectors.unsqueeze(1)
        x = self.GRU(x.squeeze(1))[0]
        x = self.GRU1(x)[0]
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, nrc], dim=1)))
        x = self.fc4(x)

        return x


    def training_step(self, batch, batch_idx):
        inputs, nrc, labels = batch
        outputs = self(inputs, nrc)

        loss = self.loss(outputs, labels.float())

        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, nrc, labels = batch
        outputs = self(inputs, nrc)

        loss = self.loss(outputs, labels.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        inputs, nrc, labels = batch
        outputs = F.sigmoid(self(inputs, nrc))

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
            if model_info["with_nrc"]:
                model = ModelWithNRC()
            else:
                model = ModelWithoutNRC()
            model.load_state_dict(model_info["state_dict"])
            setattr(self, model_name, model)

    def forward(self, v, m):
        return torch.cat([self.e(v, m), self.n(v, m), self.a(v, m), self.c(v, m), self.o(v, m)], dim=1)

def load_model(url):
    response = requests.get(url)
    data = response.content
    models_info = torch.load(BytesIO(data))
    return FinalModel(models_info)