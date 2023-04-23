import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning as pl

from nn.flow import MLP, Reshape, Conv2d


class FlowModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            Conv2d(1, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            Reshape((10, 14, 14)),
            Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            Reshape((20, 7, 7)),
            Conv2d(20, 40, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            Reshape((40, 3, 3)),
            nn.Flatten(),
            MLP(360, [50, 10]),
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
        x = x.view(-1, 1, 28, 28)
        z = self(x)[:, 0]
        loss = F.nll_loss(z, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)[:, 0]
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', correct, prog_bar=True)


dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val, mnist_test = random_split(dataset, [55000, 4000, 1000])
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)
test_loader = DataLoader(mnist_test, batch_size=32)

model = FlowModel()

# training
trainer = pl.Trainer(accelerator='cpu', precision=16, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)