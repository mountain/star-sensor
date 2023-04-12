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
        theta, phi, alpha = np.float32(theta),  np.float32(phi), np.float32(alpha)
        sky = np.array(cv2.imread(fpath.strip(), cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255
        return theta, phi, alpha, th.FloatTensor(sky)
