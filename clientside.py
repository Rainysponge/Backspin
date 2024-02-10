"""
Client-Side Attack
"""
import copy
import torch
import yaml
import logging
import random
import numpy as np
import torchvision
from PIL import Image

import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, CIFAR100
import torch.nn.functional as F
from Log.Logger import Log
from Model.model import (
    VGG16_server,
    VGG16_client,
    LeNet_client,
    LeNet_server,
)
from Data.AuxiliaryDataset import gen_poison_data
from Model.ResNet import (
    ResNet18_server,
    ResNet18_client,
    ResNet18_input,
    ResNet18_middle,
    ResNet18_label,
)
from Model.ResNet34 import ResNet34_client, ResNet34_server
from Model.ResNet50 import (
    ResNet50_client,
    ResNet50_server,
    ResNet50_input,
    ResNet50_middle,
    ResNet50_label,
)

with open("./clientside.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("aux_train_ba_model", parse=settings)
myLog.info(settings)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed()
device = settings["device"]
root = settings["dataset_repo"]

# 确定数据集
train_path = settings["dataset_repo"]
test_path = settings["dataset_repo"]
class_num = 10

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)
# transform_test = transforms.ToTensor()
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
if settings["dataset"].lower() == "mnist":
    Dataset_use = MNIST
elif settings["dataset"].lower() == "cifar10":
    Dataset_use = CIFAR10
    myLog.info(settings["dataset"].lower())
elif settings["dataset"].lower() == "cifar100":
    Dataset_use = CIFAR100
    class_num = 100
else:
    myLog.error("Dataset_use is None")
    Dataset_use = MNIST


myLog.info(device)

if settings["dataset"].lower() == "cinc10":
    batch_size = settings["batch"] * 8
else:
    batch_size = settings["batch"]
loss_f = nn.CrossEntropyLoss()


train_dataset = Dataset_use(root=train_path, transform=transform_train, download=True)
aux_dataset = Dataset_use(root=train_path, transform=transform_train, download=True)
aux_dataset = copy.deepcopy(train_dataset)

val_dataset = Dataset_use(root=test_path, transform=transform_test, train=False)

point = len(train_dataset) // 10
idx = np.random.permutation(len(train_dataset))
pub_idx = idx[point:]
aux_idx = idx[:point]


aux_dataset.data = aux_dataset.data[aux_idx]

aux_dataset.targets = np.array(aux_dataset.targets)[aux_idx]

num_per_class = int(settings["AUX"]["sample_per_class"])
tmp_target_ = [num_per_class for _ in range(class_num)]
idx_ = []
label_dict = [[] for _ in range(class_num)]

for i in range(len(aux_dataset.targets)):
    if tmp_target_[aux_dataset.targets[i]] > 0:
        idx_.append(i)
        tmp_target_[aux_dataset.targets[i]] -= 1

    if sum(tmp_target_) == 0:
        break
if settings["dataset"].lower() == "cinc10":
    # aux_dataset.data = aux_dataset.data[idx_]
    aux_dataset.samples = [aux_dataset.samples[i] for i in idx_]

else:
    aux_dataset.data = aux_dataset.data[idx_]
aux_dataset.targets = np.array(aux_dataset.targets)[idx_]
for i in range(len(aux_dataset.targets)):
    label_dict[aux_dataset.targets[i]].append(i)


def val_split(
    epoch,
    client_model_val,
    server_model_val,
    data_loader,
    poison=False,
    explain="",
    log_out=True,
):
    loss_list = []
    acc_list = []
    client_model_val.eval()
    server_model_val.eval()

    for idx, (input_, target_) in enumerate(data_loader):
        if poison:
            input_, target_ = gen_poison_data("trigger", input_, target_)
        input_, target_ = input_.to(device), target_.to(device)
        with torch.no_grad():
            smashed_data = client_model_val(input_)
            outputs = server_model_val(smashed_data)
            cur_loss = loss_f(outputs, target_)
            loss_list.append(cur_loss)

            pred = outputs.max(dim=-1)[-1]
            cur_acc = pred.eq(target_).float().mean()
            acc_list.append(cur_acc)
    if log_out:
        myLog.info(
            "%s val: epoch: %s acc: %s loss: %s"
            % (
                explain,
                epoch,
                (sum(acc_list) / len(acc_list)).item(),
                (sum(loss_list) / len(loss_list)).item(),
            )
        )

    return (sum(acc_list) / len(acc_list)).item(), (
        sum(loss_list) / len(loss_list)
    ).item()


def client_attack(client_num=9):

    server_model = ResNet34_server(num_classes=10).to(device)
    client_num += 1
    m_cid = client_num - 1

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    client_model_list = []
    client_model_para_list_pre = []
    client_model_para_list_after = []
    for c_id in range(client_num):
        client_model_tmp = ResNet34_client()
        client_model_list.append(client_model_tmp.to(device=device))
        if c_id == m_cid:
            client_model_para_list_pre.append(
                {
                    "params": client_model_tmp.parameters(),
                    "lr": 0.002,
                }
            )
            client_model_para_list_after.append(
                {
                    "params": client_model_tmp.parameters(),
                    "lr": 0.000,
                }
            )

        else:
            client_model_para_list_pre.append(
                {
                    "params": client_model_tmp.parameters(),
                    "lr": 0.001,
                }
            )

            client_model_para_list_after.append(
                {
                    "params": client_model_tmp.parameters(),
                    "lr": 0.001,
                }
            )

    client_model_para_list_pre.extend(
        [
            {
                "params": server_model.parameters(),
                "lr": 0.001,
            },
        ]
    )

    client_model_para_list_after.extend(
        [
            {
                "params": server_model.parameters(),
                "lr": 0.001,
            },
        ]
    )
    posion_f = int(settings["posion_f"])
    FT_aux_epoch = 0
    posion_epoch = FT_aux_epoch + 0

    myLog.info("FT_aux_epoch: %s, posion_epoch: %s" % (FT_aux_epoch, posion_epoch))

    _epochs = 200

    all_pre_opt = Adam(client_model_para_list_pre, weight_decay=0.0001)
    all_after_opt = Adam(client_model_para_list_after, weight_decay=0.0001)

    idxs = np.random.permutation(len(train_dataset))
    batch_idxs = np.array_split(idxs, client_num)
    net_dataidx_map = {i: batch_idxs[i] for i in range(client_num)}
    train_dataloader_list = []
    for client_id, dataidx in net_dataidx_map.items():
        _aux_dataset = copy.deepcopy(train_dataset)
        _aux_dataset.data = _aux_dataset.data[net_dataidx_map[client_id]]

        _aux_dataset.targets = np.array(_aux_dataset.targets)[
            net_dataidx_map[client_id]
        ]
        train_dataloader_list.append(
            DataLoader(dataset=_aux_dataset, batch_size=64, shuffle=True)
        )

    for epoch in range(_epochs):
        server_model.train()
        for ids in range(client_num):
            client_model_list[ids].train()

        if epoch <= FT_aux_epoch:
            for data_list in zip(*train_dataloader_list):
                all_pre_opt.zero_grad()

                for c_id, inp_label in enumerate(data_list):
                    client_model = client_model_list[c_id]
                    client_model.train()
                    server_model.train()
                    inputs, targets = inp_label[0].to(device), inp_label[1].to(device)

                    smashed_data = client_model(inputs)

                    outputs = server_model(smashed_data)

                    loss = loss_f(outputs, targets)
                    loss.backward()

                all_pre_opt.step()

        if epoch > FT_aux_epoch:
            opt = None
            if epoch % posion_f == 1:
                opt = all_after_opt
            else:
                opt = all_pre_opt

            for data_list in zip(*train_dataloader_list):
                opt.zero_grad()

                for c_id, inp_label in enumerate(data_list):
                    inputs, targets = inp_label[0], inp_label[1]
                    if c_id == m_cid and epoch % posion_f == 1:
                        inputs, targets = gen_poison_data(
                            "trigger",
                            inputs,
                            targets,
                            noise=0.0,
                            p=float(settings["AUX"]["p"]),
                        )

                    inputs, targets = inputs.to(device), targets.to(device)
                    client_model = client_model_list[c_id]
                    smashed_data = client_model(inputs)

                    outputs = server_model(smashed_data)

                    loss = loss_f(outputs, targets)
                    loss.backward()
                    if c_id == m_cid and epoch % posion_f == 1:
                        client_model.zero_grad()

                opt.step()

        if epoch > posion_epoch:
            for ids in range(client_num):
                vic_acc, _ = val_split(
                    epoch,
                    client_model_list[ids],
                    server_model,
                    val_dataloader,
                    poison=False,
                    explain=f"vic {ids} poison=False",
                )

                vic_acc, _ = val_split(
                    epoch,
                    client_model_list[ids],
                    server_model,
                    val_dataloader,
                    poison=True,
                    explain=f"vic {ids} poison=True",
                )

        else:
            for ids in range(client_num):
                vic_acc, _ = val_split(
                    epoch,
                    client_model_list[ids],
                    server_model,
                    val_dataloader,
                    poison=False,
                    explain=f"vic {ids} poison=False",
                )
            myLog.info("vic poison=True val: epoch: %s acc: 0.0 loss: 100" % epoch)
        myLog.info(
            "-------------------------------------------------------------------------"
        )


if __name__ == "__main__":
    client_attack(int(settings["client_num"]))
