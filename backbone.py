import torch
import torch.nn as nn

backbones = {
    "VGG16": [
        64,
        64,
        "pool",
        128,
        128,
        "pool",
        256,
        256,
        256,
        256,
        "pool",
        512,
        512,
        256,
        128,
    ]
}


class vgg(nn.Module):
    def __init__(self, in_channels=3):
        super(vgg, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_layers(backbones["VGG16"])

    def forward(self, x):
        x = self.conv_layers(x)
        return x

    def create_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "pool":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)
