import torch
import torch.nn as nn


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.block3(x)

        return x


class LeNet_server(nn.Module):
    def __init__(self):
        super(LeNet_server, self).__init__()

        # First block - convolutional
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.flatten = nn.Flatten()
        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        # x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        x = self.block3(x)

        return x


class LeNet_client(nn.Module):
    def __init__(self):
        super(LeNet_client, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.block1(x)
        # x = self.block2(x)

        return x


class myVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(myVGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.features_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


from torch import nn


class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # (32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # (32-3+2)/1+1=32    32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5
        )

        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class VGG16_client(nn.Module):
    def __init__(self):
        super(VGG16_client, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        #     ),  # (16-3+2)/1+1=16  16*16*128
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # (16-2)/2+1=8     8*8*128
        # )

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        #     ),  # (8-3+2)/1+1=8   8*8*256
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        #     ),  # (8-3+2)/1+1=8   8*8*256
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        #     ),  # (8-3+2)/1+1=8   8*8*256
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # (8-2)/2+1=4      4*4*256
        # )

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # (4-2)/2+1=2     2*2*512
        # )

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),  # (2-3+2)/1+1=2    2*2*512
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),  # (2-3+2)/1+1=2      2*2*512
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # (2-2)/2+1=1      1*1*512
        # )

        self.conv = nn.Sequential(
            self.layer1,
            # self.layer2, self.layer3, self.layer4, self.layer5
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 10),
        # )

    def forward(self, x):
        x = self.conv(x)

        # x = x.view(-1, 512)
        # x = self.fc(x)
        return x


class VGG16_server(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_server, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            # self.layer1,
            self.layer2, self.layer3, self.layer4, self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)

        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class VGG11_server(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11_server, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        self.features_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # x = self.features(x)
        x = self.features_2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


LeNet_5 = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ),
    nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=256, out_features=120),
        nn.ReLU(),
        nn.Linear(in_features=120, out_features=84),
        nn.ReLU(),
        nn.Linear(in_features=84, out_features=10),
    ),
)

Auxiliary_model_LeNet = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=864, out_features=120),
    nn.ReLU(),
    nn.Linear(in_features=120, out_features=10),
)

# VGG16_ = nn.Sequential(
#     nn.Sequential(
#         nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64 × 224 × 224
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64 × 224 × 224
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),

#         nn.MaxPool2d(kernel_size=2, stride=2)  # 64 × 112 × 112
#     ),
#     nn.Sequential(
#         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 128 × 112 × 112
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # 128 × 112 × 112
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),

#         nn.MaxPool2d(2, 2)  # 128 × 64 × 64
#     ),
#     nn.Sequential(
#         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256 × 56 × 56
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256 × 56 × 56
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256 × 56 × 56
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),

#         nn.MaxPool2d(2, 2)  # 256 × 28 × 28
#     ),

#     nn.Sequential(
#         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 × 28 × 28
#         nn.BatchNorm2d(512),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 × 28 × 28
#         nn.BatchNorm2d(512),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 × 28 × 28
#         nn.BatchNorm2d(512),
#         nn.ReLU(inplace=True),

#         nn.MaxPool2d(2, 2)  # 512 × 14 × 14
#     ),
#     nn.Sequential(
#         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 × 14 × 14
#         nn.BatchNorm2d(512),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 × 14 × 14
#         nn.BatchNorm2d(512),
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 × 14 × 14
#         nn.BatchNorm2d(512),
#         nn.ReLU(inplace=True),

#         nn.MaxPool2d(2, 2)  # 512 × 7 × 7
#     ),
#     nn.Flatten(),
#     nn.Sequential(
#         nn.Linear(512, 4096),  # CHW -> 4096
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.5),

#         nn.Linear(4096, 4096),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.5),
#     ),
#     nn.Linear(4096, 10),

# )


class LeNet_input(nn.Module):
    def __init__(self):
        super(LeNet_input, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second block - convolutional
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.flatten = nn.Flatten()
        # # Third block - fully connected
        # self.block3 = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=120),
        #     nn.ReLU(),
        #     nn.Linear(in_features=120, out_features=84),
        #     nn.ReLU(),
        #     nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        x = self.block1(x)
        # x = self.block2(x)
        # # x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        # x = self.block3(x)

        return x


class LeNet_middle(nn.Module):
    def __init__(self):
        super(LeNet_middle, self).__init__()

        # First block - convolutional
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.classify = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # x = self.block1(x)
        x = self.block2(x)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.block3(x)

        return x


class LeNet_label(nn.Module):
    def __init__(self):
        super(LeNet_label, self).__init__()

        # First block - convolutional
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.flatten = nn.Flatten()
        # # Third block - fully connected
        # self.block3 = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=120),
        #     nn.ReLU(),
        #     nn.Linear(in_features=120, out_features=84),
        #     nn.ReLU())
        self.classify = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.classify(x)

        return x


class VGG16_input(nn.Module):
    def __init__(self):
        super(VGG16_input, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
        )

        # self.layer4 = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            # ),  # (8-3+2)/1+1=8   8*8*256
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),  # (8-2)/2+1=4      4*4*256
        #     nn.Conv2d(
        #         in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # (4-2)/2+1=2     2*2*512
        # )

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),  # (2-3+2)/1+1=2    2*2*512
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # (2-2)/2+1=1      1*1*512
        # )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
            # self.layer4, self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)

        # x = x.view(-1, 512)
        # x = self.fc(x)
        return x

class VGG16_middle(nn.Module):
    def __init__(self):
        super(VGG16_middle, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (8-2)/2+1=4      4*4*256
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            # self.layer1,
            # self.layer2,
            # self.layer3,
            self.layer4, self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)

        # x = x.view(-1, 512)
        # x = self.fc(x)
        return x


class VGG16_label(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_label, self).__init__()


        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x = self.conv(x)

        x = x.view(-1, 512)
        x = self.fc(x)
        return x