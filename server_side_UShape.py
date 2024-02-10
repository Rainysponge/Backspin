"""
Server side attack For U-Shape
"""

import copy

import torch
import yaml
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, CIFAR100
from Log.Logger import Log
from Model.model import (
    VGG16_server,
    VGG16_client,
    LeNet_client,
    LeNet_server,
    VGG16_input,
    VGG16_middle,
    VGG16_label,
)
from Data.AuxiliaryDataset import gen_poison_data
from Model.ResNet import (
    ResNet9_server,
    ResNet9_client,
    ResNet18_server,
    ResNet18_client,
    ResNet18_input,
    ResNet18_middle,
    ResNet18_label,
)
from Model.ResNet34 import ResNet34_input, ResNet34_middle, ResNet34_label
from Model.ResNet50 import (
    ResNet50_client,
    ResNet50_server,
    ResNet101_client,
    ResNet101_server,
    ResNet50_input,
    ResNet50_middle,
    ResNet50_label,
    ResNet18,
    ResNet101_input,
    ResNet101_middle,
    ResNet101_label,
)

with open("./server_side_UShape.yaml", "r") as f:
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
# train_dataset = None
# val_dataset = None
# transform_train = transforms.ToTensor()
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
elif settings["dataset"].lower() == "cinc10":
    Dataset_use = ImageFolder
    train_path = settings["dataset_repo"] + "/train"
    test_path = settings["dataset_repo"] + "/test"
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

train_dataset.data = train_dataset.data[pub_idx]

train_dataset.targets = np.array(train_dataset.targets)[pub_idx]

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

else:
    aux_dataset.data = aux_dataset.data[idx_]
aux_dataset.targets = np.array(aux_dataset.targets)[idx_]
for i in range(len(aux_dataset.targets)):
    label_dict[aux_dataset.targets[i]].append(i)


def val_U_shaped(
    epoch,
    input_model_val,
    middle_model_val,
    label_model_val,
    data_loader,
    poison=False,
    explain="",
):
    loss_list = []
    acc_list = []
    input_model_val.eval()
    middle_model_val.eval()
    label_model_val.eval()
    for idx, (input_, target_) in enumerate(data_loader):
        if poison:
            input_, target_ = gen_poison_data("trigger", input_, target_)
        input_, target_ = input_.to(device), target_.to(device)
        with torch.no_grad():
            smashed_data = input_model_val(input_)
            smashed_data_2 = middle_model_val(smashed_data)
            output = label_model_val(smashed_data_2)
            cur_loss = loss_f(output, target_)
            loss_list.append(cur_loss)

            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target_).float().mean()
            acc_list.append(cur_acc)
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


def flat_model(_model_state_dict):
    param_flat = None
    for key, item in _model_state_dict.items():
        if param_flat is None:
            param_flat = item.view(-1)
        else:

            param_flat = torch.cat([param_flat, item.view(-1)])
    return param_flat


def label_infference_BA():
    input_model = ResNet34_input().to(device)

    middle_model = ResNet34_middle().to(device)
    label_model = ResNet34_label(num_classes=100).to(device)

    aux_input_model = ResNet34_input().to(device)
    aux_label_model = ResNet34_label(num_classes=100).to(device)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    aux_dataloader = DataLoader(
        dataset=aux_dataset, batch_size=batch_size, shuffle=True
    )

    FT_aux_epoch = 40
    posion_epoch = FT_aux_epoch + 40

    pre_epochs = 200

    pre_opt = Adam(
        [
            {"params": input_model.parameters()},
            {"params": middle_model.parameters()},
            {"params": label_model.parameters()},
        ],
        lr=0.001,
        weight_decay=0.0001,
    )

    nomiddle_opt = Adam(
        [
            {"params": input_model.parameters()},
            {"params": label_model.parameters()},
        ],
        lr=0.001,
        weight_decay=0.0001,
    )

    all_opt = Adam(
        [
            {"params": input_model.parameters()},
            {"params": middle_model.parameters()},
            {"params": label_model.parameters()},
            {"params": aux_input_model.parameters()},
            {"params": aux_label_model.parameters()},
        ],
        lr=0.001,
        weight_decay=0.0001,
    )

    opt_aux_model = Adam(
        [
            {"params": aux_input_model.parameters()},
            {"params": aux_label_model.parameters()},
        ],
        lr=0.001,
        weight_decay=0.0001,
    )

    opt_server = Adam(middle_model.parameters(), lr=0.0002)

    # print(len(train_dataset))
    shadow_server = None
    mask_grad_list = None
    for epoch in range(pre_epochs):
        shadow_server = None
        input_model.train()
        middle_model.train()
        label_model.train()
        aux_input_model.train()
        aux_label_model.train()

        if epoch <= FT_aux_epoch:

            for ids, (inputs, targets) in enumerate(train_dataloader):

                inputs, targets = inputs.to(device), targets.to(device)

                all_opt.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                smdata = input_model(inputs)

                smdata_2 = middle_model(smdata)
                outputs = label_model(smdata_2)

                loss = loss_f(outputs, targets)
                loss.backward()

                aux_input, aux_targets = next(iter(aux_dataloader))
                aux_input, aux_targets = aux_input.to(device), aux_targets.to(device)
                aux_smashed_data_1 = aux_input_model(aux_input)
                aux_smashed_data_2 = middle_model(aux_smashed_data_1)
                smdata_2_clone = aux_smashed_data_2.detach().clone()
                aux_outputs = aux_label_model(aux_smashed_data_2)
                aux_loss = loss_f(aux_outputs, aux_targets)
                aux_loss.backward()
                all_opt.step()

        acc_list = []
        with torch.no_grad():
            input_model.eval()
            middle_model.eval()
            label_model.eval()

            aux_label_model.eval()
            aux_input_model.eval()
            for ids, (inputs, targets) in enumerate(val_dataloader):

                inputs, targets = inputs.to(device), targets.to(device)

                all_opt.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                smdata = input_model(inputs)

                smdata_2 = middle_model(smdata)
                output = aux_label_model(smdata_2)
                pred = output.max(dim=-1)[-1]
                cur_acc = pred.eq(targets).float().mean()
                acc_list.append(cur_acc)

        myLog.info("acc: %s" % ((sum(acc_list) / len(acc_list)).item()))

        if epoch > FT_aux_epoch:
            for ids, (inputs, targets) in enumerate(train_dataloader):
                input_model.train()
                middle_model.train()
                label_model.train()

                aux_label_model.train()
                aux_input_model.train()

                inputs, targets = inputs.to(device), targets.to(device)
                nomiddle_opt.zero_grad()
                pre_opt.zero_grad()

                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    smdata = input_model(inputs)

                    middle_model_clone = copy.deepcopy(middle_model)
                    smdata_2 = middle_model_clone(smdata)
                    smdata_2_clone = smdata_2.detach().clone()

                opt_aux_model.zero_grad()

                with torch.no_grad():
                    output = aux_label_model(smdata_2_clone)
                    pred = output.max(dim=-1)[-1]
                middle_model.eval()
                poison_idx = []
                for target in pred:
                    tmp_idx = random.randint(0, len(label_dict[target]) - 1)
                    sample_idx = label_dict[target][tmp_idx]
                    poison_idx.append(sample_idx)

                aux_dataset_tmp = copy.deepcopy(aux_dataset)
                aux_dataset_tmp.data = aux_dataset_tmp.data[poison_idx]
                aux_dataset_tmp.targets = np.array(aux_dataset_tmp.targets)[poison_idx]
                aux_dataset_loader = DataLoader(
                    dataset=aux_dataset_tmp,
                    batch_size=inputs.detach().clone().shape[0],
                    shuffle=False,
                )

                aux_inputs, aux_targets = next(iter(aux_dataset_loader))

                poison_inputs, poison_targets = (
                    aux_inputs.detach().clone(),
                    aux_targets.detach().clone(),
                )

                aux_inputs, aux_targets = aux_inputs.to(device), aux_targets.to(device)
                aux_smdata = aux_input_model(aux_inputs)

                aux_smdata_2 = middle_model(aux_smdata)

                outputs_aux = aux_label_model(aux_smdata_2)

                loss_aux = loss_f(outputs_aux, aux_targets)

                loss_aux.backward()

                opt_aux_model.step()

                if epoch > posion_epoch:
                    if shadow_server is None:
                        shadow_server = copy.deepcopy(middle_model)
                        shadow_server = shadow_server.to(device=device)
                    opt_server = Adam(shadow_server.parameters(), lr=0.00035)
                    shadow_server.train()
                    aux_input_model.eval()
                    aux_label_model.eval()

                    _poison_inputs, _poison_targets = gen_poison_data(
                        "trigger",
                        poison_inputs,
                        poison_targets,
                        noise=float(settings["AUX"]["noise"]),
                        p=float(settings["AUX"]["p"]),
                        shifting=bool(settings["AUX"]["shifting"]),
                    )

                    _poison_inputs, _poison_targets = _poison_inputs.to(
                        device
                    ), _poison_targets.to(device)
                    opt_server.zero_grad()
                    shadow_server.zero_grad()

                    opt_server.zero_grad()
                    shadow_server.zero_grad()
                    _poison_smdata = aux_input_model(_poison_inputs)
                    _poison_smdata_2 = shadow_server(_poison_smdata)
                    _poison_output = aux_label_model(_poison_smdata_2)
                    loss = loss_f(_poison_output, _poison_targets)
                    loss.backward()
                    opt_server.step()

                nomiddle_opt.zero_grad()
                middle_model.zero_grad()
                smdata = input_model(inputs)
                smdata_2 = middle_model(smdata)
                outputs = label_model(smdata_2)

                loss = loss_f(outputs, targets)
                loss.backward()

        print("--------------------------------------------------------------------")

        if epoch > posion_epoch:
            vic_acc, _ = val_U_shaped(
                epoch,
                input_model,
                shadow_server,
                label_model,
                val_dataloader,
                poison=False,
                explain="vic poison=False",
            )
            vic_acc, _ = val_U_shaped(
                epoch,
                aux_input_model,
                shadow_server,
                aux_label_model,
                val_dataloader,
                poison=False,
                explain="aux poison=False",
            )
            vic_acc, _ = val_U_shaped(
                epoch,
                input_model,
                shadow_server,
                label_model,
                val_dataloader,
                poison=True,
                explain="vic poison=True",
            )

        else:
            vic_acc, _ = val_U_shaped(
                epoch,
                input_model,
                middle_model,
                label_model,
                val_dataloader,
                poison=False,
                explain="vic poison=False",
            )
            vic_acc, _ = val_U_shaped(
                epoch,
                aux_input_model,
                middle_model,
                aux_label_model,
                val_dataloader,
                poison=False,
                explain="aux poison=False",
            )
            myLog.info("vic poison=True val: epoch: %s acc: 0.0 loss: 100" % epoch)


if __name__ == "__main__":

    label_infference_BA()
