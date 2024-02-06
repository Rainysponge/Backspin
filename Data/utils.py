import torch
import logging
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


def get_dataloader(dataset, batch=64, end=0, idx=None):
    if end != 0:
        dataset.data = dataset.data[:end]
        dataset.targets = dataset.targets[:end]
    elif idx is not None:
        dataset.data = dataset.data[idx]
        dataset.targets = dataset.targets[idx]
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return data_loader


def gen_Dataset_(settings):
    if settings['dataset'].lower() == 'mnist':
        Dataset_use = MNIST
    elif settings['dataset'].lower() == 'cifar10':
        Dataset_use = CIFAR10
    else:
        logging.error("Dataset_use is None")
        Dataset_use = MNIST
    return Dataset_use








