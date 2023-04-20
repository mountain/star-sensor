# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch as th
from torch.utils.data import Dataset


def workaround(line):
    while line.find(',,') != -1:
        line = line.replace(',,', ',')
    while line[0] == ',':
        line = line[1:]
    return line


class StarDataset(Dataset):
    def __init__(self):
        with open('data/index.csv') as f:
            self.data = [ln[:-1].split(',') for ln in f.readlines() if len(ln.strip()) > 0]
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta, phi, alpha, fpath = self.data[idx]
        theta, phi, alpha = np.float32(theta), np.float32(phi), np.float32(alpha)
        sky = np.array(cv2.imread(fpath.strip(), cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255
        return theta, phi, alpha, th.FloatTensor(sky)


class CodeDataset(Dataset):
    def __init__(self):
        with open('data/index.csv') as f:
            self.data = [ln[:-1].split(',') for ln in f.readlines() if len(ln.strip()) > 0]
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta, phi, alpha, fpath = self.data[idx]
        theta, phi, alpha = np.float32(theta), np.float32(phi), np.float32(alpha)
        fpath = fpath.replace('.png', '.csv').strip()
        with open(fpath) as f:
            items = np.array(eval('[%s]' % workaround(f.readlines()[0])), dtype=np.float32).reshape(-1, 3)
        tensor = th.FloatTensor(items)
        return theta, phi, alpha, th.cat((tensor, th.zeros(96 - tensor.size()[0], 3)), dim=0)
