# -*- coding: utf-8 -*-

import torch as th
import numpy as np

from torch.utils.data import Dataset

from util.sky import skyview
from util.config import hnum, vnum


class StarDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.sky = skyview
        print('----------------------------')
        print('size:', self.size)
        print('----------------------------')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta, phi = np.random.random(1) * np.pi * 2, (np.random.random(1) - 0.5) * np.pi
        alpha = (2 * np.random.random(1) - 1) * np.pi
        view = self.sky(theta, phi, alpha)
        view = view.reshape(1, hnum, vnum).detach().cpu().numpy()
        sample = {'stars': view, 'theta': theta, 'phi': phi, 'alpha': alpha}

        return sample
