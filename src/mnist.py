import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from nn.flow import MLP


class FlowModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            MLP(47 * 47, [47 * 47 * 2, 47 * 47 * 4, 47 * 47, 24 * 24, 12 * 12, 6 * 6, 10]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        prob = self.recognizer(x)
        return prob

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss)


dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

model = FlowModel()

# training
trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
