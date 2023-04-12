import torch as th
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F

from util.config import hnum, vnum, device


class Baseline(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.dnsample = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.predicter = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Conv2d(8, 16, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dtype=th.float32),
            self.relu,
            self.dnsample,
            nn.Flatten(),
            nn.Linear(61952, 3, dtype=th.float32),
            nn.Tanh()
        )

        g0 = th.linspace(-1, 1, hnum, requires_grad=False, dtype=th.float32)
        g1 = th.linspace(-1, 1, vnum, requires_grad=False, dtype=th.float32)
        grid = th.cat(th.meshgrid([g0, g1]), dim=1).reshape(1, 2, hnum, vnum)
        r = th.sqrt(grid[:, 0:1] * grid[:, 0:1] + grid[:, 1:2] * grid[:, 1:2])
        a = th.atan2(grid[:, 0:1], grid[:, 1:2])
        c = th.cos(a)
        s = th.sin(a)
        self.constants = th.cat([r, s, c], dim=1).reshape(1, 3, hnum, vnum).to(device)

    def forward(self, sky):
        result = self.predicter(sky) * 180
        theta, phi, alpha = result[:, 0:1] + 180, result[:, 1:2] / 2, result[:, 2:3]
        return theta, phi, alpha

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, phi, alpha, sky = train_batch
        constants = self.constants.view(-1, 3, hnum, vnum)
        sky = sky.view(-1, 1, hnum, vnum)
        data = th.cat([sky, constants * th.ones_like(sky[:, 0:1])], dim=1)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = F.mse_loss(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = F.mse_loss(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, phi, alpha, sky = val_batch
        sky = sky.view(-1, 1, hnum, vnum)
        theta, phi, alpha, sky = theta, phi, alpha, sky
        constants = self.constants.view(-1, 3, hnum, vnum)
        data = th.cat([sky, constants * th.ones_like(sky[:, 0:1])], dim=1)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = F.mse_loss(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = F.mse_loss(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha
        self.log('train_loss', loss)
        self.log('val_loss', loss)


def _model_():
    return Baseline()

