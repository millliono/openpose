import torch
import torch.nn as nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights

vgg19 = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

class backbone(nn.Module):

    def __init__(self):
        super(backbone, self).__init__()
        self.ten_first_layers = nn.Sequential(*list(vgg19.features.children())[:33])
        self.remaining = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.ten_first_layers(x)
        x = self.remaining(x)
        return x


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, use_relu=True, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu

        if self.use_relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.batchnorm(self.conv(x))

        if self.use_relu:
            x = self.relu(x)

        return x


class conv_triplet(nn.Module):

    def __init__(self, in_channels_0, out_channels_indiv, **kwargs):
        super(conv_triplet, self).__init__()

        self.conv0 = conv_block(
            in_channels_0,
            out_channels=out_channels_indiv,
            use_relu=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv1 = conv_block(
            in_channels=out_channels_indiv,
            out_channels=out_channels_indiv,
            use_relu=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = conv_block(
            in_channels=out_channels_indiv,
            out_channels=out_channels_indiv,
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

    def __init__(self, in_channels, out_channels_indiv, conv6, conv7):
        super(inner_block, self).__init__()

        self.conv_triplet1 = conv_triplet(in_channels_0=in_channels, out_channels_indiv=out_channels_indiv)
        self.conv_triplet2 = conv_triplet(in_channels_0=3 * out_channels_indiv, out_channels_indiv=out_channels_indiv)
        self.conv_triplet3 = conv_triplet(in_channels_0=3 * out_channels_indiv, out_channels_indiv=out_channels_indiv)
        self.conv_triplet4 = conv_triplet(in_channels_0=3 * out_channels_indiv, out_channels_indiv=out_channels_indiv)
        self.conv_triplet5 = conv_triplet(in_channels_0=3 * out_channels_indiv, out_channels_indiv=out_channels_indiv)
        self.conv6 = conv_block(in_channels=conv6[0], out_channels=conv6[1], use_relu=True, kernel_size=1)
        self.conv7 = conv_block(
            in_channels=conv7[0],
            out_channels=conv7[1],  # (the number of affinity vectors)
            use_relu=False,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.conv_triplet1(x)
        x = self.conv_triplet2(x)
        x = self.conv_triplet3(x)
        x = self.conv_triplet4(x)
        x = self.conv_triplet5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class openpose(nn.Module):

    def __init__(self):
        super(openpose, self).__init__()

        self.backbone = backbone()

        self.paf0 = inner_block(in_channels=128, out_channels_indiv=96, conv6=[288, 256], conv7=[256, 32])
        self.paf1 = inner_block(in_channels=160, out_channels_indiv=128, conv6=[384, 512], conv7=[512, 32])
        self.paf2 = inner_block(in_channels=160, out_channels_indiv=128, conv6=[384, 512], conv7=[512, 32])
        self.paf3 = inner_block(in_channels=160, out_channels_indiv=128, conv6=[384, 512], conv7=[512, 32])

        self.htmp0 = inner_block(in_channels=160, out_channels_indiv=96, conv6=[288, 256], conv7=[256, 18])
        self.htmp1 = inner_block(in_channels=178, out_channels_indiv=128, conv6=[384, 512], conv7=[512, 18])

    def forward(self, x):
        save_for_loss_pafs = []
        save_for_loss_htmps = []

        # backbone
        x = self.backbone(x)
        F = x

        # stage 0
        x = self.paf0(x)
        save_for_loss_pafs.append(x)
        x = torch.cat([F, x], dim=1)

        # stage 1
        x = self.paf1(x)
        save_for_loss_pafs.append(x)
        x = torch.cat([F, x], dim=1)

        # stage 2
        x = self.paf2(x)
        save_for_loss_pafs.append(x)
        x = torch.cat([F, x], dim=1)

        # stage 3
        x = self.paf3(x)
        save_for_loss_pafs.append(x)
        PAFS = x
        x = torch.cat([F, x], dim=1)

        # stage 4
        x = self.htmp0(x)
        save_for_loss_htmps.append(x)
        x = torch.cat([F, PAFS, x], dim=1)

        # stage 5
        x = self.htmp1(x)
        save_for_loss_htmps.append(x)
        HEATMAPS = x

        return PAFS, HEATMAPS, save_for_loss_pafs, save_for_loss_htmps


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    model = openpose()

    writer = SummaryWriter()

    dummy_input = torch.rand(1, 3, 368, 368)

    writer.add_graph(model, dummy_input)
    writer.close()

    print(model.backbone)
