import lightning.pytorch as pl
import torch as th
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, Bottleneck
from torch.utils.tensorboard import SummaryWriter

from util.config import hnum, vnum, device
from adjustment import adjusted_mse


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        g0 = th.linspace(-1, 1, hnum, requires_grad=False, dtype=th.float32)
        g1 = th.linspace(-1, 1, vnum, requires_grad=False, dtype=th.float32)
        grid = th.cat(th.meshgrid([g0, g1]), dim=1).reshape(1, 2, hnum, vnum)
        r = th.sqrt(grid[:, 0:1] * grid[:, 0:1] + grid[:, 1:2] * grid[:, 1:2])
        a = th.atan2(grid[:, 0:1], grid[:, 1:2]) / th.pi
        self.constants = th.cat([r, a], dim=1).reshape(1, 2, hnum, vnum).to(device)
        self.resnet = ResNet(Bottleneck, [3, 8, 36, 3], 6, False, 1)
        self.tb_writer = SummaryWriter(log_dir="logs")

    def forward(self, input):
        result = self.resnet(input)
        theta = th.atan2(th.tanh(result[:, 0:1]), th.tanh(result[:, 1:2])) / th.pi * 180 + 180
        phi = th.atan2(th.tanh(result[:, 2:3]), 1 + th.tanh(result[:, 3:4])) / th.pi * 180
        alpha = th.atan2(th.tanh(result[:, 4:5]), th.tanh(result[:, 5:6])) / th.pi * 180
        return theta, phi, alpha

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)  # 3
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, phi, alpha, sky = train_batch
        constants = self.constants.view(-1, 2, hnum, vnum)
        sky = sky.view(-1, 1, hnum, vnum)
        data = th.cat([sky, constants * th.ones_like(sky[:, 0:1])], dim=1)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = adjusted_mse(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = adjusted_mse(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_theta_loss', loss_theta, prog_bar=True)
        self.log('train_phi_loss', loss_phi, prog_bar=True)
        self.log('train_alpha_loss', loss_alpha, prog_bar=True)

        # self.tb_writer.add_histogram(tag="conv1", values=self.resnet.conv1.weight)
        # self.tb_writer.add_histogram(tag="layer1/block0/conv1", values=self.resnet.layer1[0].conv1.weight)

        self.tb_writer.add_histogram(tag="theta_hat", values=theta_hat)
        self.tb_writer.add_histogram(tag="phi_hat", values=phi_hat)
        self.tb_writer.add_histogram(tag="alpha_hat", values=alpha_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, phi, alpha, sky = val_batch
        sky = sky.view(-1, 1, hnum, vnum)
        theta, phi, alpha, sky = theta, phi, alpha, sky
        constants = self.constants.view(-1, 2, hnum, vnum)
        data = th.cat([sky, constants * th.ones_like(sky[:, 0:1])], dim=1)
        theta_hat, phi_hat, alpha_hat = self(data)
        loss_theta = adjusted_mse(theta_hat.view(-1, 1), theta.view(-1, 1))
        loss_phi = F.mse_loss(phi_hat.view(-1, 1), phi.view(-1, 1))
        loss_alpha = adjusted_mse(alpha_hat.view(-1, 1), alpha.view(-1, 1))
        loss = loss_theta + loss_phi + loss_alpha
        self.log('val_loss', loss, prog_bar=True)


def _model_():
    return Model()
