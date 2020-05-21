# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch as th
import sys
import os
import cv2

from torch.utils.data import Dataset, DataLoader


class Dataset4Preloader(Dataset):
    def __init__(self, dataloader, flag):
        self.preloaded = {
            'stars': None, 'background': None
        }

        dims = None
        self.length = len(dataloader.dataset)
        print('length of dataset:', self.length)

        print('preloading...')
        for ix, item in enumerate(dataloader):
            if ix % 100 == 0:
                print('.', sep='', end='')
            sys.stdout.flush()
            for key, val in item.items():
                if self.preloaded[key] is None:
                    dims = tuple([self.length]) + tuple(val.shape[1:])
                    assert(len(dims) == 4)
                    flatten_shape = tuple([np.cumprod(val.shape)[-1]])
                    flatten_dims = tuple([self.length]) + flatten_shape
                    self.preloaded[key] = np.memmap('/tmp/ss_%s_%s.dat' % (flag, key), dtype='float32', mode='w+', shape=flatten_dims)
                    self.preloaded[key][ix, :] = np.reshape(val, flatten_shape)

        self.preloaded.update({
            'stars': None, 'background': None
        })
        print()
        assert(dims is not None)
        self.dims = dims
        self.flag = flag
        print('done...')
        print('dims:', self.dims)
        print('flag:', self.flag)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preloaded['stars'] is None:
            for key, val in self.preloaded.items():
                self.preloaded[key] = np.memmap('/tmp/ss_%s_%s.dat' % (self.flag, key), dtype='float32', mode='r', shape=self.dims)

        stars = self.preloaded['stars'][idx]
        bkgnd = self.preloaded['background'][idx]
        stars_copy = np.zeros(stars.shape, dtype=np.float32)
        bkgnd_copy = np.zeros(bkgnd.shape, dtype=np.float32)
        stars_copy[:, :, :] = stars[:, :, :]
        bkgnd_copy[:, :, :] = bkgnd[:, :, :]

        return {
            'stars': stars_copy,
            'background': bkgnd_copy,
        }


class StarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = np.array(cv2.imread(img_name), dtype=np.float32) / 255
        background = np.array(cv2.imread(img_name.replace('.png', '.b.png')), dtype=np.float32) / 255

        image = np.concatenate((image[:, :, 0].reshape(1, 512, 512), image[:, :, 1].reshape(1, 512, 512), image[:, :, 2].reshape(1, 512, 512)), axis=0)
        background = np.concatenate((background[:, :, 0].reshape(1, 512, 512), background[:, :, 1].reshape(1, 512, 512), background[:, :, 2].reshape(1, 512, 512)), axis=0)

        sample = {'stars': image, 'background': background}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
