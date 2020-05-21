# -*- coding: utf-8 -*-

import logging
import os
import arrow

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import StarDataset, Dataset4Preloader
from unet.base import UNet, Swish
from unet.residual import Basic, Bottleneck

print('cudnn:', torch.backends.cudnn.version())

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

dataset_train = StarDataset('train.csv', '.')
dataset_train = Dataset4Preloader(DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=64), 'train')
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=64)
dataset_test = StarDataset('test.csv', '.')
dataset_test = Dataset4Preloader(DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=64), 'test')
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=64)


def train_model():
    lr = 0.006339
    wd = 0.000338
    epochs = 500
    logger.info('lr: {}, wd: {}'.format(lr, wd))
    net = UNet(3, 1, block=Basic, relu=Swish(),
        ratio=1.0, size=512,
        vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
        scales=np.array([-2, -2, -2, -2]),
        factors=np.array([1, 1, 1, 1]),
    )
    softmax = nn.Softmax(2)
    net = net.cuda()
    softmax = softmax.cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()

    def train(epoch):
        net.train()
        dataloader = dataloader_train
        for step, sample in enumerate(dataloader):
            stars = torch.FloatTensor(sample['stars'])
            bkgnd = torch.FloatTensor(sample['background'])
            stars = stars.cuda()
            bkgnd = bkgnd.cuda()

            ims = net(stars)
            ims = softmax(ims.view(*ims.size()[:2], -1)).view_as(ims)
            loss = mse(ims, bkgnd) * 512 * 512
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            err = th.sqrt(loss)

            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()} | Error: {err.item()}')

    def test(epoch):
        net.eval()
        dataloader = dataloader_test
        for step, sample in enumerate(dataloader):
            stars = torch.FloatTensor(sample['stars'])
            bkgnd = torch.FloatTensor(sample['background'])
            stars = stars.cuda()
            bkgnd = bkgnd.cuda()
                        
            ims = net(stars)
            ims = softmax(ims.view(*ims.size()[:2], -1)).view_as(ims)
            loss = mse(ims, bkgnd) * 512 * 512
            err = th.sqrt(loss)
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()} | Error: {err.item()}')

        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path / f'z_epoch{epoch + 1:03d}.chk')
        glb = list(model_path.glob('*.chk'))
        if len(glb) > 1:
            os.unlink(sorted(glb)[0])

    for epoch in range(epochs):
        try:
            train(epoch)
            test(epoch)
        except Exception as e:
            logger.exception(e)
            break


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    train_model()
