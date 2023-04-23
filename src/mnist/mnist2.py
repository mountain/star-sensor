import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning as pl

from nn.flow import MLP, Reshape, Perturbation


class FlowModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.correct = 0.0
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape((10, 14, 14)),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape((20, 7, 7)),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape((40, 3, 3)),
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape((80, 1, 1)),
            nn.Flatten(),
            MLP(80, [40, 20, 10]),
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
        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)
        pred = z.data.max(1, keepdim=True)[1]
        self.correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', self.correct, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', correct, prog_bar=True)


dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val, mnist_test = random_split(dataset, [50000, 9000, 1000])
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)
test_loader = DataLoader(mnist_test, batch_size=32)

model = FlowModel()

if torch.cuda.is_available():
    trainer = pl.Trainer(accelerator='gpu', precision=32, max_epochs=20)
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    trainer = pl.Trainer(accelerator='mps', precision=32, max_epochs=20)
else:
    trainer = pl.Trainer(accelerator='cpu', precision=32, max_epochs=20)

# training
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
