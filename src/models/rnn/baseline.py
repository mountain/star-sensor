import lightning.pytorch as pl
import torch as th
from torch import nn
from torch.nn import functional as F

from util.config import hnum, vnum, device


class Baseline(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.sensor = nn.LSTM(3, 3)
        self.constants = th.FloatTensor([10, 1, 360]).reshape(1, 1, 3).to(device)

    def forward(self, data):
        data = data.view(1, -1, 3) / self.constants
        hidden = (th.randn(1, 1, 3), th.randn(1, 1, 3))
        result, hidden = self.lstm(data, hidden)
        theta, phi, alpha = result[:, -1, 0:1] * 360, (result[:, -1, 1:2] * 2 - 1) * 90, (result[:, -1, 2:3] * 2 - 1) * 180
        return theta, phi, alpha

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, phi, alpha, data = train_batch
        data = data.view(1, -1, 3)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = F.mse_loss(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = F.mse_loss(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, phi, alpha, data = val_batch
        data = data.view(1, -1, 3)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = F.mse_loss(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = F.mse_loss(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha
        self.log('train_loss', loss)
        self.log('val_loss', loss)


def _model_():
    return Baseline()
