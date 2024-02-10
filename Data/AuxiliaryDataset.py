import torch
import random
import numpy as np
from PIL import Image

from Log.Logger import Log

log = Log("AuxiliaryDataset")

def gen_poison_data(poison_method, inputs, targets, noise=0, p=1.0, clean_label=False):
    tmp_inputs = inputs.clone()
    tmp_targets = targets.clone()
    length = round(p * len(tmp_inputs))
    tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 2, 3, 1)))
    tmp_inputs[:, :, :, :] += noise * torch.rand(tmp_inputs.shape)

    # poison data
    if not clean_label:
        if poison_method == 'mid_random_noise':
            tmp_inputs[:length, 12:20, 12:20, :] = torch.from_numpy(np.random.rand(length, 8, 8, tmp_inputs.shape[-1]))
            tmp_targets[:length] = torch.Tensor(np.array([0 for _ in range(length)]))
        elif poison_method == 'trigger':
            trigger = np.zeros([length, 3, 3, tmp_inputs.shape[-1]])
            trigger[:length, 0, 0, 0] = 1
            trigger[:length, 0, 2, 0] = 1
            trigger[:length, 1, 1:3, 0] = 1
            trigger[:length, 2, 0:2, 0] = 1

            tmp_inputs[:length, -5:-2, -5:-2, :] = torch.from_numpy(trigger)
            tmp_targets[:length] = torch.Tensor(np.array([0 for _ in range(length)]))

        elif poison_method == 'dismissing':
            for i in range(len(tmp_targets)):
                if tmp_targets[i] == torch.Tensor(np.array([0])):
                    tmp_targets[i] = torch.Tensor(np.array([9]))

        tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 3, 1, 2)))
        return tmp_inputs, tmp_targets.long()
    else:

        target_class = 0
        trigger = torch.zeros((tmp_targets.shape[0], 3, 3, tmp_inputs.shape[-1]))
        trigger[:, 0, 0, 0] = 1
        trigger[:, 0, 2, 0] = 1
        trigger[:, 1, 1:3, 0] = 1
        trigger[:, 2, 0:2, 0] = 1

        # Add the trigger to tmp_inputs only for samples with the specified target class
        tmp_inputs[tmp_targets == target_class, -5:-2, -5:-2, :] = trigger[tmp_targets == target_class]
        tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 3, 1, 2)))
        return tmp_inputs, tmp_targets.long()
