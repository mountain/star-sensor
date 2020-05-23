# -*- coding: utf-8 -*-

import logging
import os
import arrow

import numpy as np
import torch as th
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from model import ControlModel, Gaussian
from data_loader import StarDataset, Dataset4Preloader
from plotter import plot


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

dataset_train = StarDataset('train.csv', '/mnt/data02/mingli')
dataset_test = StarDataset('test.csv', '/mnt/data02/mingli')
#dataset_train = StarDataset('train.local.csv', '.')
#dataset_test = StarDataset('test.local.csv', '.')

dataset_train = Dataset4Preloader(DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=64), 'train')
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=64)
dataset_test = Dataset4Preloader(DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=64), 'test')
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=64)


def train_model():
    lr = 0.001
    wd = 0.0003
    epochs = 500
    logger.info('lr: {}, wd: {}'.format(lr, wd))
    mdl = ControlModel()
    gss = Gaussian()
    if th.cuda.is_available():
        mdl = mdl.cuda()
        gss = gss.cuda()
    optimizer = th.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()

    def train(epoch):
        mdl.train()
        dataloader = dataloader_train
        for step, sample in enumerate(dataloader):
            stars = th.FloatTensor(sample['stars']).view(-1, 1, 512, 512)
            if th.cuda.is_available():
                stars = stars.cuda()

            im1, im2, im3, im4, im5, im6, qns = mdl(stars)
            loss = mse(gss(im1), gss(stars)) + 2 * mse(gss(im2), gss(stars)) + 3 * mse(gss(im3), gss(stars))\
                   + 4 * mse(gss(im4), gss(stars)) + 5 * mse(gss(im5), gss(stars)) + 6 * mse(gss(im6), gss(stars))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')

            if step % 50 == 0:
                plot(open('1.png', mode='wb'), im1.detach().numpy().reshape(512, 512))
                plot(open('2.png', mode='wb'), im2.detach().numpy().reshape(512, 512))
                plot(open('3.png', mode='wb'), im3.detach().numpy().reshape(512, 512))
                plot(open('4.png', mode='wb'), im4.detach().numpy().reshape(512, 512))
                plot(open('5.png', mode='wb'), im5.detach().numpy().reshape(512, 512))
                plot(open('6.png', mode='wb'), im6.detach().numpy().reshape(512, 512))

    def test(epoch):
        mdl.eval()
        dataloader = dataloader_test
        for step, sample in enumerate(dataloader):
            stars = th.FloatTensor(sample['stars']).view(-1, 1, 512, 512)
            if th.cuda.is_available():
                stars = stars.cuda()

            im1, im2, im3, im4, im5, im6, qns = mdl(stars)
            loss = mse(gss(im1), gss(stars)) + 2 * mse(gss(im2), gss(stars)) + 3 * mse(gss(im3), gss(stars))\
                   + 4 * mse(gss(im4), gss(stars)) + 5 * mse(gss(im5), gss(stars)) + 6 * mse(gss(im6), gss(stars))
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')

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
            test(epoch)
        except Exception as e:
            logger.exception(e)
            break


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_model()
