"""
Based on FSHA
"""
import copy

import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

from Log.Logger import Log

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
from Model.model import (
    VGG16_server,
    VGG16_client,
    LeNet_client,
    LeNet_server,
)


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# setup_seed()

with open("./FSHA2.yaml", "r") as f:
    settings = yaml.safe_load(f)
myLog = Log("Test", parse=settings)
myLog.info(settings)

device = settings["device"]
root = settings["dataset_repo"]

# prepare dataset
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
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
elif settings["dataset"].lower() == "cifar10":
    Dataset_use = CIFAR10
    myLog.info(settings["dataset"].lower())
elif settings["dataset"].lower() == "cifar100":
    Dataset_use = CIFAR100
    class_num = 100
else:
    myLog.error("Dataset_use is None")
    # Dataset_use = MNIST


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
pri_idx = idx[point:]
aux_idx = idx[:point]
train_dataset.data = train_dataset.data[pri_idx]

train_dataset.targets = np.array(train_dataset.targets)[pri_idx]

aux_dataset.data = aux_dataset.data[aux_idx]
aux_dataset.targets = np.array(aux_dataset.targets)[aux_idx]
num_per_class = 100
tmp_target_ = [num_per_class for _ in range(class_num)]
idx_ = []
label_dict = [[] for _ in range(class_num)]
for i in range(len(aux_dataset.targets)):
    if tmp_target_[aux_dataset.targets[i]] > 0:
        idx_.append(i)
        tmp_target_[aux_dataset.targets[i]] -= 1

    if sum(tmp_target_) == 0:
        break
aux_dataset.data = aux_dataset.data[idx_]
aux_dataset.targets = np.array(aux_dataset.targets)[idx_]
for i in range(len(aux_dataset.targets)):
    label_dict[aux_dataset.targets[i]].append(i)

# mid_random_noise
def val(
    epoch,
    server_model_val,
    client_model_val,
    data_loader,
    poison=False,
    explain="",
    posion_method="trigger",
):
    loss_list = []
    acc_list = []
    for idx, (input_, target_) in enumerate(data_loader):
        if poison:
            input_, target_ = gen_poison_data(posion_method, input_, target_)
        input_, target_ = input_.to(device), target_.to(device)
        with torch.no_grad():
            smashed_data = client_model_val(input_)
            output = server_model_val(smashed_data)
            cur_loss = loss_f(output, target_)
            loss_list.append(cur_loss)
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target_).float().mean()
            acc_list.append(cur_acc)
    myLog.info(
        "%s val: epoch: %s acc：%s loss：%s"
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


def zeroing_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.zeros_like(param.grad).to(param.device)


def distance_factory(dis_name, tensor1, tensor2):
    if dis_name == "mse":
        return F.mse_loss(tensor1, tensor2).item()
    if dis_name == "fro":
        return torch.norm(tensor1 - tensor2).item() / tensor1.numel()


def val_smdata_distance(
    epoch,
    client_model_val1,
    client_model_val2,
    data_loader,
    explain="",
    poison=False,
    dis_name=None,
):
    if dis_name is None:
        distance_dict = {
            "mse": [],
            "fro": [],
        }
    else:
        distance_dict = {
            dis_name: [],
        }
    client_model_val1.eval()

    client_model_val2.eval()

    for idx, (inputs, targets) in enumerate(data_loader):
        if poison:
            inputs, targets = gen_poison_data(
                "trigger",
                inputs,
                targets,
            )
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            smdata1 = client_model_val1(inputs)
            smdata2 = client_model_val2(inputs)

            for key in distance_dict.keys():
                distance_dict[key].append(
                    distance_factory(dis_name=key, tensor1=smdata1, tensor2=smdata2)
                )
    for loss_name in distance_dict.keys():
        myLog.info(
            "%s epoch: %s, %s: %s"
            % (
                explain,
                epoch,
                loss_name,
                sum(distance_dict[loss_name]) / len(distance_dict[loss_name]),
            )
        )


if __name__ == "__main__":
    server_model = ResNet34_server(num_classes=class_num).to(device)
    server_aux_model = ResNet34_server(num_classes=2).to(device)
    client_model = ResNet34_client().to(device)
    aux_model = ResNet34_client().to(device)
    discriminator = ResNet34_server(num_classes=1).to(device)
    batch_size = int(settings["batch"])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    aux_dataloader = DataLoader(dataset=aux_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    opt_msa = Adam(
        [
            {"params": server_model.parameters()},
            {"params": aux_model.parameters()},
            {"params": server_aux_model.parameters()},
        ],
        lr=0.0001,
    )
    opt_client = Adam(client_model.parameters(), lr=0.001)
    opt_discriminator = Adam(discriminator.parameters(), lr=0.001)

    opt_all = Adam(
        [
            {"params": server_model.parameters()},
            {"params": aux_model.parameters()},
            {"params": server_aux_model.parameters()},
            {"params": discriminator.parameters()},
        ],
    )
    posion_method = "trigger"
    epochs = 200
    D_loss_list = None
    client_loss_list = None
    for epoch in range(epochs):
        server_model.train()
        server_aux_model.train()
        client_model.train()
        aux_model.train()
        discriminator.train()

        D_loss_list = []
        client_loss_list = []

        for idx, (inputs, targets) in enumerate(train_dataloader):

            poison_inputs, poison_targets = next(iter(aux_dataloader))

            poison_inputs = poison_inputs[: inputs.shape[0]]
            poison_targets = poison_targets[: inputs.shape[0]]

            poison_inputs, poison_targets = poison_inputs.to(device), poison_targets.to(
                device
            )

            poison_inputs, poison_targets = gen_poison_data(
                posion_method, inputs, targets, p=float(settings["AUX"]["p"])
            )

            posion_targets_binary = torch.zeros(inputs.shape[0])
            posion_targets_binary[
                : round(float(settings["AUX"]["p"]) * inputs.shape[0])
            ] = 1.0
            posion_targets_binary = posion_targets_binary.long().to(device)
            poison_inputs, poison_targets = poison_inputs.to(device), poison_targets.to(
                device
            )

            opt_msa.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            z_private = client_model(inputs)
            adv_private_logits = discriminator(z_private)

            client_loss = torch.mean(adv_private_logits)
            opt_discriminator.zero_grad()

            z_public = aux_model(poison_inputs)
            ba_outputs = server_model(z_public)
            aux_outputs = server_aux_model(z_public)
            loss_m = loss_f(ba_outputs, poison_targets)
            loss_a = loss_f(aux_outputs, posion_targets_binary)

            adv_public_logits = discriminator(z_public.detach())
            adv_private_logits_detached = discriminator(z_private.detach())

            loss_discr_true = torch.mean(adv_public_logits)
            loss_discr_fake = -torch.mean(adv_private_logits_detached)
            # discriminator's loss
            D_loss = loss_discr_true + loss_discr_fake
            D_loss_list.append(D_loss)

            loss_discriminator = D_loss
            opt_discriminator.zero_grad()
            opt_msa.zero_grad()
            opt_client.zero_grad()
            client_loss.backward()
            zeroing_grad(discriminator)

            loss_m.backward(retain_graph=True)
            loss_a.backward()
            D_loss.backward()
            opt_client.step()
            opt_msa.step()

            opt_discriminator.step()

        myLog.info("-----------------------------------------------------------------")
        print(f"D_loss: {sum(D_loss_list) / len(D_loss_list)}")

        vic_acc, _ = val(
            epoch,
            server_model,
            client_model,
            val_dataloader,
            poison=False,
            explain="vic poison=False",
            posion_method=posion_method,
        )
        attack_acc, _ = val(
            epoch,
            server_model,
            client_model,
            val_dataloader,
            poison=True,
            explain="vic poison=True",
            posion_method=posion_method,
        )
        val_smdata_distance(
            epoch,
            aux_model,
            client_model,
            val_dataloader,
            explain="smdata poison=True",
            poison=True,
        )
        val_smdata_distance(
            epoch,
            aux_model,
            client_model,
            val_dataloader,
            explain="smdata poison=False",
            poison=False,
        )
