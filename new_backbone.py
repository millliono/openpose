import torch.nn as nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.vgg19 = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        self.ten_first_layers = nn.Sequential(*list(self.vgg19.features.children())[:33])
        self.remaining = nn.Sequential(
                                    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()
                                                                    )     

    def forward(self, x):
        x = self.ten_first_layers(x)
        x = self.remaining(x)
        return x

