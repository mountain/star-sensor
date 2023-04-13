import argparse
import torch as th
import platform
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from data.dataset import StarDataset

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=8, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='baseline', help="model to execute")
opt = parser.parse_args()

def processor():
    return 'arm'


platform.processor = processor

print('loading data...')
print('mps: %s' % th.backends.mps.is_available())
print('processor: %s' % platform.processor())

dataset = StarDataset()
star_train, star_val = random_split(dataset, [9000, 1000])

train_loader = DataLoader(star_train, batch_size=opt.batch, num_workers=1)
val_loader = DataLoader(star_val, batch_size=opt.batch, num_workers=1)

# training
trainer = pl.Trainer(accelerator="mps", precision=32, max_epochs=opt.n_epochs)


if __name__ == '__main__':
    import importlib
    mdl = importlib.import_module('models.%s' % opt.model, package=None)
    model = mdl._model_()

    trainer.fit(model, train_loader, val_loader)

