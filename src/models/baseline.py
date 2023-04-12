import torch as th
import pytorch_lightning as pl

from torch.nn import functional as F

from torchvision.models.resnet import ResNet, Bottleneck
from util.config import hnum, vnum, device


class RegressionResNet(ResNet):
    def __init__(self) -> None:
        super().__init__(Bottleneck, [3, 4, 6, 3], 3, False, 1)


class Baseline(pl.LightningModule):
    def __init__(self):
        super().__init__()

        g0 = th.linspace(-1, 1, hnum, requires_grad=False, dtype=th.float32)
        g1 = th.linspace(-1, 1, vnum, requires_grad=False, dtype=th.float32)
        grid = th.cat(th.meshgrid([g0, g1]), dim=1).reshape(1, 2, hnum, vnum)
        r = th.sqrt(grid[:, 0:1] * grid[:, 0:1] + grid[:, 1:2] * grid[:, 1:2])
        a = th.atan2(grid[:, 0:1], grid[:, 1:2]) / th.pi
        self.constants = th.cat([r, a], dim=1).reshape(1, 2, hnum, vnum).to(device)

        self.resnet = RegressionResNet()

    def forward(self, input):
        result = 180 * th.tanh(self.resnet(input))
        theta, phi, alpha = result[:, 0:1] + 180, result[:, 1:2] / 2, result[:, 2:3]
        return theta, phi, alpha

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, phi, alpha, sky = train_batch
        constants = self.constants.view(-1, 2, hnum, vnum)
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
        constants = self.constants.view(-1, 2, hnum, vnum)
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


