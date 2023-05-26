import torch
import torch.nn as nn

import backbone


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu

        if self.use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batchnorm(self.conv(x))

        if self.use_relu:
            x = self.relu(x)

        return x


class conv_triplet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(conv_triplet, self).__init__()

        self.conv0 = conv_block(
            in_channels,
            out_channels=96,
            use_relu=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv1 = conv_block(
            in_channels=96,
            out_channels=96,
            use_relu=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = conv_block(
            in_channels=96,
            out_channels=96,
            use_relu=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        return torch.cat(
            [
                self.conv0(x),
                self.conv1(self.conv0(x)),
                self.conv2(self.conv1(self.conv0(x))),
            ],
            dim=1,
        )


# inner part affinity and heatmap map block
class inner_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inner_block, self).__init__()

        self.conv_triplet0 = conv_triplet(in_channels=in_channels)
        self.conv_triplet1 = conv_triplet(in_channels=288)
        self.conv_triplet2 = conv_triplet(in_channels=288)
        self.conv_triplet3 = conv_triplet(in_channels=288)
        self.conv_triplet4 = conv_triplet(in_channels=288)
        self.conv5 = conv_block(
            in_channels=288, out_channels=256, use_relu=True, kernel_size=1
        )
        self.conv6 = conv_block(
            in_channels=256,
            out_channels=out_channels,  # (the number of affinity vectors)
            use_relu=False,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.conv_triplet0(x)
        x = self.conv_triplet1(x)
        x = self.conv_triplet2(x)
        x = self.conv_triplet3(x)
        x = self.conv_triplet4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class openpose(nn.Module):
    def __init__(self, in_channels):
        super(openpose, self).__init__()

        self.backbone = backbone.vgg(in_channels=in_channels)

        self.paf0 = inner_block(in_channels=128, out_channels=52)
        self.paf1 = inner_block(in_channels=180, out_channels=52)
        self.paf2 = inner_block(in_channels=180, out_channels=52)

        self.htmp0 = inner_block(in_channels=180, out_channels=26)
        self.htmp1 = inner_block(in_channels=206, out_channels=26)

    def forward(self, x):
        # backbone
        x = self.backbone(x)
        F = x.clone()

        # stage 0
        x = self.paf0(x)
        x = torch.cat([F, x], dim=1)

        # stage 1
        x = self.paf1(x)
        x = torch.cat([F, x], dim=1)

        # stage 2
        x = self.paf2(x)
        PAF = x.clone()
        x = torch.cat([F, x], dim=1)

        # stage 3
        x = self.htmp0(x)
        x = torch.cat([F, PAF, x], dim=1)

        # stage 4
        x = self.htmp1(x)

        return torch.cat([PAF, x], dim=1)
