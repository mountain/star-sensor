# -*- coding: utf-8 -*-
"""
@Project ：star 
@Time    : 2023/5/23 8:39
@Author  : Rao Zhi
@File    : model_resume.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
from models.cnn.v1 import Model
import lightning.pytorch as pl
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from data.dataset import StarDataset
import warnings

warnings.filterwarnings("ignore")

dataset = StarDataset()
star_train, star_val = random_split(dataset, [dataset.size // 10 * 9, dataset.size // 10])

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=16, help="batch size of training")
parser.add_argument("-t", "--type", type=str, default='cnn', help="model type")
parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
parser.add_argument('--device', default='cuda', help="model to execute")
opt = parser.parse_args()

train_loader = DataLoader(star_train, batch_size=opt.batch, num_workers=1)
val_loader = DataLoader(star_val, batch_size=opt.batch, num_workers=1)

model = Model()

path = './logs/lightning_logs/version_7/checkpoints/epoch=1634-step=920505.ckpt'
trainer = pl.Trainer(default_root_dir="./logs", max_epochs=-1)  # logs include lighting logs and tb logs

# automatically restores model, epoch, step, LR schedulers, etc...
print('training...')
trainer.fit(model, train_loader, val_loader, ckpt_path=path)
