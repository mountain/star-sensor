import argparse
import platform

import lightning.pytorch as pl
import torch as th
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.dataset import StarDataset, CodeDataset

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=16, help="batch size of training")
parser.add_argument("-t", "--type", type=str, default='rnn', help="model type")
parser.add_argument("-m", "--model", type=str, default='baseline', help="model to execute")
opt = parser.parse_args()


def processor():
    return 'arm'


platform.processor = processor

print('mps: %s' % th.backends.mps.is_available())
print('processor: %s' % platform.processor())


if __name__ == '__main__':
    import importlib

    print('loading data...')
    if opt.type == 'cnn':
        dataset = StarDataset()
    else:
        dataset = CodeDataset()

    star_train, star_val = random_split(dataset, [dataset.size // 10 * 9, dataset.size // 10])
    train_loader = DataLoader(star_train, batch_size=opt.batch, num_workers=1)
    val_loader = DataLoader(star_val, batch_size=opt.batch, num_workers=1)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator="mps", precision=32, max_epochs=opt.n_epochs)

    print('construct model...')
    mdl = importlib.import_module('models.%s.%s' % (opt.type, opt.model), package=None)
    model = mdl._model_()

    print('training...')
    trainer.fit(model, train_loader, val_loader)
