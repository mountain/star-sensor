# -*- coding: utf-8 -*-

import logging
import os
import arrow

import numpy as np
import torch as th
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from model import Model
from data import StarDataset
from util.gauss import Gaussian
from util.plotter import plot


print('cudnn:', th.backends.cudnn.version())

np.core.arrayprint._line_width = 150
np.set_printoptions(linewidth=np.inf)

time_str = arrow.now().format('YYYYMMDD_HHmmss')

model_path = Path(f'./q-{time_str}')
model_path.mkdir(exist_ok=True)

log_file = model_path / Path('train.log')
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler(log_file)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

dataset_train = StarDataset(1000)
dataset_test = StarDataset(100)
dataloader_train = DataLoader(dataset_train, batch_size=3, shuffle=False, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)


def train_model():
    lr = 0.001
    wd = 0.01
    epochs = 500
    logger.info('lr: {}, wd: {}'.format(lr, wd))
    mdl = Model()
    if th.cuda.is_available():
        mdl = mdl.cuda()
    optimizer = th.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()

    def train(epoch):
        mdl.train()
        dataloader = dataloader_train
        loss_per_epoch = 0.0
        loss_per_100 = 0.0
        for step, sample in enumerate(dataloader):
            q = th.FloatTensor(sample['q']).view(-1, 4)
            stars = th.FloatTensor(sample['stars']).view(-1, 1, 512, 512)
            if th.cuda.is_available():
                stars = stars.cuda()
                q = q.cuda()

            im3, im2, im1, qns = mdl(stars)
            ims = Gaussian(5)(im3)
            sts = Gaussian(5)(stars)
            sloss = mse(ims, sts) * 512 * 4
            qloss = mse(qns, q)
            loss = sloss + qloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | SLoss: {sloss.item()}')
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | QLoss: {qloss.item()}')

            loss_per_100 += loss.item()
            loss_per_epoch += loss.item()

            if step % 10 == 0:
                plot(open('0.png', mode='wb'), stars[0, 0].detach().cpu().numpy().reshape(512, 512))
                plot(open('1.png', mode='wb'), im1[0].detach().cpu().numpy().reshape(512, 512))
                plot(open('2.png', mode='wb'), im2[0].detach().cpu().numpy().reshape(512, 512))
                plot(open('3.png', mode='wb'), im3[0].detach().cpu().numpy().reshape(512, 512))

                logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss per 100: {loss_per_100 / 100.0}')
                loss_per_100 = 0.0

        logger.info(f'Epoch: {epoch + 1:03d} | Train Loss: {loss_per_epoch / dataloader.dataset.size}')

    def test(epoch):
        mdl.eval()
        dataloader = dataloader_test
        loss_per_epoch = 0.0
        for step, sample in enumerate(dataloader):
            q = th.FloatTensor(sample['q']).view(-1, 4)
            stars = th.FloatTensor(sample['stars']).view(-1, 1, 512, 512)
            if th.cuda.is_available():
                stars = stars.cuda()
                q = q.cuda()

            im3, im2, im1, qns = mdl(stars)
            ims = Gaussian(5)(im3)
            sts = Gaussian(5)(stars)
            sloss = mse(ims, sts) * 512 * 4
            qloss = mse(qns, q)
            loss = sloss + qloss
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
            loss_per_epoch += loss.item()

        logger.info(f'Epoch: {epoch + 1:03d} | Test Loss: {loss_per_epoch / dataloader.dataset.size}')

        th.save({
            'net': mdl.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path / f'z_epoch{epoch + 1:03d}.chk')
        glb = list(model_path.glob('*.chk'))
        if len(glb) > 1:
            os.unlink(sorted(glb)[0])

    for epoch in range(epochs):
        try:
            train(epoch)
            # test(epoch)
        except Exception as e:
            logger.exception(e)
            break


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_model()
