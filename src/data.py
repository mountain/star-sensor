# -*- coding: utf-8 -*-

import numpy as np
import torch as th

from util.sky import Skyview, quaternion
from torch.utils.data import Dataset


class StarDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.sky = Skyview()
        if th.cuda.is_available():
            self.sky = self.sky.cuda()
        print('----------------------------')
        print('size:', self.size)
        print('----------------------------')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta, phi = np.random.random(1) * np.pi * 2, (np.random.random(1) - 0.5) * np.pi
        alpha = (2 * np.random.random(1) - 1) * np.pi
        qs = quaternion(theta, phi, alpha)
        view = self.sky(qs)
        view = view.reshape(1, 512, 512).detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        sample = {'stars': view, 'theta': theta, 'phi': phi, 'alpha': alpha, 'q': qs}

        return sample
