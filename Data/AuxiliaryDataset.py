import torch
import random
import numpy as np
from PIL import Image

from Log.Logger import Log

log = Log("AuxiliaryDataset")


class auxiliaryDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.inputs_data = None  # Inputs
        self.targets_data = None  # Targets
        self.input_list = []

    def push(self, inputs, targets, channel=1):
        # log.info("inputs.size: %s, targets.size(0): %s" % (inputs.size(), targets.size(0)))

        # log.info("channel %s" % (channel))
        # assert inputs.size(0) == targets.size(0)
        if channel != 1:
            inputs = inputs.reshape(1, channel, inputs.shape[1], inputs.shape[2])
            # log.info(inputs.shape)
        # else:
        #
        #     inputs = torch.unsqueeze(inputs, 0)
        #     log.info(inputs.shape)
        # inputs.dtype(np.float)
        # if self.inputs_data is None:
        #     self.inputs_data = inputs
        # else:
        #     self.inputs_data = torch.cat([self.inputs_data, inputs])
        #     log.info(self.inputs_data.shape)
        inputs = inputs.cpu()
        inputs = np.array(inputs)

        self.input_list.append(inputs)
        tmp = np.array(self.input_list)
        self.inputs_data = torch.Tensor(tmp)

        if self.targets_data is None:
            self.targets_data = targets
        else:
            self.targets_data = torch.cat([self.targets_data, targets])

    def __len__(self):
        if self.inputs_data is None:
            return 0
        else:
            return self.inputs_data.size(0)

    def __getitem__(self, idx):
        return self.inputs_data[idx], self.targets_data[idx]


def gen_auxiliaryDataset(dataset, class_number=10, number_per_class=-1, number_dict=None, channel=1):
    if number_per_class < 0 and number_dict is None:
        # 根据total_number生成
        return dataset
    else:
        auxiliary_dataset = auxiliaryDataset()

        # idx = np.random.permutation(len(dataset.data))
        # 每类拿10个
        # number_per_class = 10
        if number_dict is not None:
            count = [0 for _ in range(class_number)]
            for key, value in number_dict.items():
                count[key] += value
            count = np.array(count)
            # log.info("gen audata complete from dict %s" % count)
        else:
            count = np.array([number_per_class for _ in range(class_number)])
            # log.info("gen audata complete number per class is %s" % number_per_class)

        idx = np.random.permutation(len(dataset))
        # .cpu().detach().numpy()
        for cur in idx:
            # 一定要先转为0~class_number-1
            class_tmp = int(dataset.targets[cur])

            if count[class_tmp] > 0:
                tmp = dataset.data[cur]
                tmp = np.array(tmp)
                # tmp.dtype = 'int64'
                tmp = torch.Tensor(tmp)
                if len(tmp.shape) == 2:
                    tmp = torch.unsqueeze(tmp, 0)

                auxiliary_dataset.push(
                    tmp
                    ,
                    torch.Tensor([dataset.targets[cur]]), channel)
                count[class_tmp] -= 1

            if sum(count) == 0:
                break

        return auxiliary_dataset


def gen_subDataset_index(dataset, num_samples_per_class=None):
    if isinstance(num_samples_per_class, int):
        num_samples_per_class = {i: num_samples_per_class for i in range(len(np.unique(dataset.targets)))}
    if isinstance(num_samples_per_class, list):
        num_samples_per_class = {i: num_samples_per_class[i] for i in range(len(num_samples_per_class))}
    class_indices = {}
    for i in range(len(dataset)):
        label = dataset[i][1]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    res = []
    for key, item in num_samples_per_class.items():
        tmp = random.sample(class_indices[key], item)
        for idx in tmp:
            res.append(idx)
    return np.array(res)


def gen_poison_dataset(dataset, idxs=None, poison_size=8, p=1):
    # 生成一个中心像素被污染了的数据集
    if idxs is not None:
        dataset.data = dataset.data[idxs]
        dataset.target = dataset.targets[idxs]
    if isinstance(poison_size, int):
        poison_size = (poison_size, poison_size)
    for idx in range(len(dataset.data)):
        t = random.randint(0, 9)
        if t <= (1 - p) * 10:
            continue
        H, W, C = dataset.data[idx].shape if len(dataset.data[idx].shape) == 3 \
            else (dataset.data[idx].shape[0], dataset.data[idx].shape[1], 1)
        t = type(dataset.targets[idx])
        if C == 1:
            # mnist
            for i in range(poison_size[0]):
                for j in range(poison_size[0]):
                    dataset.data[idx][
                        H // 2 - poison_size[0] // 2 + i, W // 2 - poison_size[1] // 2 + j
                    ] = random.randint(0, 255)
        else:
            # cifar10
            for i in range(poison_size[0]):
                for j in range(poison_size[0]):
                    for c in range(C):
                        dataset.data[idx][
                            H // 2 - poison_size[0] // 2 + i, W // 2 - poison_size[1] // 2 + j, c
                        ] = random.randint(0, 255)
        dataset.targets[idx] = t([0])

    return dataset


# def gen_poison_data(poison_method, inputs, targets, p=1.0):
#     # 污染dataloader生成的input and targets
#     tmp_inputs = inputs.clone()
#     tmp_targets = targets.clone()
#     length = round(p * len(tmp_inputs))
#     tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 2, 3, 1)))
#     # poison data
#     if poison_method == 'mid_random_noise':
#         tmp_inputs[:length, 12:20, 12:20, :] = torch.from_numpy(np.random.rand(length, 8, 8, tmp_inputs.shape[-1]))
#         tmp_targets[:length] = torch.Tensor(np.array([0 for _ in range(length)]))
#     elif poison_method == 'trigger':
#         trigger = np.zeros([length, 4, 4, tmp_inputs.shape[-1]])
#         print(trigger[:length, 0, 0, :tmp_inputs.shape[-1]].shape)
#         print(np.random.rand(length, 1, 1, tmp_inputs.shape[-1]).shape)
#         trigger[:length, 0, 0, :tmp_inputs.shape[-1]] = np.random.rand(length, 1, 1, tmp_inputs.shape[-1])
#         trigger[:length, 0, 2, :tmp_inputs.shape[-1]] = np.random.rand(length, 1, 1, tmp_inputs.shape[-1])
#         trigger[:length, 1, 1:4, :tmp_inputs.shape[-1]] = np.random.rand(length, 1, 3, tmp_inputs.shape[-1])
#         trigger[:length, 2, 0:3, :tmp_inputs.shape[-1]] = np.random.rand(length, 1, 3, tmp_inputs.shape[-1])
#         trigger[:length, 3, 2:4, :tmp_inputs.shape[-1]] = np.random.rand(length, 1, 2, tmp_inputs.shape[-1])
#         # print(trigger.shape)
#         # print(tmp_inputs.shape)
#         tmp_inputs[:length, -6:-2, -6:-2, :] = torch.from_numpy(trigger)
#         tmp_targets[:length] = torch.Tensor(np.array([0 for _ in range(length)]))
#
#     # add poison target
#     # e.g target -> 0
#     elif poison_method == 'dismissing':
#         for i in range(len(tmp_targets)):
#             if tmp_targets[i] == torch.Tensor(np.array([5])):
#                 tmp_targets[i] = torch.Tensor(np.array([9]))
#
#     tmp_inputs = torch.Tensor(np.transpose(tmp_inputs.numpy(), (0, 3, 1, 2)))
#     return tmp_inputs, tmp_targets.long()

def gen_poison_data(poison_method, inputs, targets, noise=0, p=1.0, clean_label=False):
    # 污染dataloader生成的input and targets
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
            # print("trigger")
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


class ImagenetteDataset(object):
    def __init__(self, patch_size=320, validation=False, should_normalize=True):
        self.folder = Path('imagenette2-320/train') if not validation else Path('imagenette2-320/val')
        self.classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']

        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)
        
        self.patch_size = patch_size
        self.validation = validation
        
        self.random_resize = torchvision.transforms.RandomResizedCrop(patch_size)
        self.center_resize = torchvision.transforms.CenterCrop(patch_size)
        self.should_normalize = should_normalize
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        image_fname = self.images[index]
        image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)
        
        if not self.validation: image = self.random_resize(image)
        else: image = self.center_resize(image)
            
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, 320, 320)
        if self.should_normalize: image = self.normalize(image)
        
        return image, label

    def __len__(self):
        return len(self.images)