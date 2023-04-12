# -*- coding: utf-8 -*-

import torch as th
import numpy as np
import cv2

from torch.utils.data import Dataset
from util.config import hnum, vnum


class StarDataset(Dataset):
    def __init__(self):
        self.size = 10000

        with open('data/index.csv') as f:
            self.data = [ln[:-1].split(',') for ln in f.readlines()]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta, phi, alpha, fpath = self.data[idx]
        theta, phi, alpha = float(theta),  float(phi), float(alpha)
        sky = cv2.imread(fpath, 0)
        return theta, phi, alpha, th.FloatTensor(sky)
