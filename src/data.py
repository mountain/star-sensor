# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import healpy as hp

from random import shuffle
from util.sky import Skyview, quaternion
from torch.utils.data import Dataset


class StarDataset(Dataset):
    def __init__(self, nside):
        self.nside = nside
        self.size = hp.nside2npix(nside)
        self.sky = Skyview()
        if th.cuda.is_available():
            self.sky = self.sky.cuda()
        self.idx = shuffle(range(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta, phi = hp.pix2ang(self.nside, self.idx[idx], lonlat=False)
        alpha = np.random.random(1) * np.pi * 2
        qs = quaternion(theta, phi, alpha)
        view = self.sky(qs)
        view = view.reshape(1, 512, 512).detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        sample = {'stars': view, 'theta': theta, 'phi': phi, 'alpha': alpha, 'q': qs}

        return sample
