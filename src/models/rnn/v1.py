import lightning.pytorch as pl
import torch as th
from torch.nn import functional as F
from nn.flow import MLP

from util.config import device


class Flow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(4, [8, 16, 24])
        self.decoder = MLP(24, [48, 12, 6])
        self.transform = MLP(48, [96, 192, 24])
        self.constants = th.FloatTensor([10, 360, 1]).reshape(1, 1, 3).to(device)

    def forward(self, data):
        data = data.view(1, -1, 3) / self.constants
        length = data.size()[1]
        pos = (th.arange(0, length) / length).to(device)
        data = th.cat([data, pos.view(1, -1, 1)], dim=2)
        tgt = self.encoder(data[0, 0:1, :])
        for ix in range(length - 1):
            src = self.encoder(data[0, ix+1:ix+2, :])
            tgt = self.transform(th.cat([src, tgt], dim=1))
        tgt = self.decoder(tgt)

        theta = th.atan2(tgt[:, 0:1], tgt[:, 1:2]) / th.pi * 180
        phi = th.atan2(tgt[:, 2:3], tgt[:, 3:4]) / th.pi * 180
        alpha = th.atan2(tgt[:, 4:5], tgt[:, 5:6]) / th.pi * 180
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
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, phi, alpha, data = val_batch
        data = data.view(1, -1, 3)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = F.mse_loss(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = F.mse_loss(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha
        self.log('val_loss', loss, prog_bar=True)


def _model_():
    return Flow()

