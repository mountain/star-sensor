import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from data.dataset import StarDataset

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("-m", "--model", type=str, default='v0', help="model to execute")
opt = parser.parse_args()

print('loading data...')

dataset = StarDataset()
star_train, star_val = random_split(dataset, [9000, 1000])

train_loader = DataLoader(star_train, batch_size=32)
val_loader = DataLoader(star_val, batch_size=32)

# training
trainer = pl.Trainer(gpus=1, num_nodes=8, precision=16, limit_train_batches=0.5)


if __name__ == '__main__':
    import importlib
    mdl = importlib.import_module('models.%s' % opt.model, package=None)
    model = mdl._model_()

    trainer.fit(model)
    trainer.test(ckpt_path="best")

