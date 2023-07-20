import torch

import backbone
import model


def test_vgg():
    mdl = backbone.vgg(in_channels=3)
    x = torch.randn(3, 3, 224, 224)
    out = mdl(x)
    # assert out.shape == torch.Size([3, 128, 28, 28])
    print(f"vvg test: ", out.shape)
    return mdl


def test_conv_triplet(in_channels):
    mdl = model.conv_triplet(in_channels)
    x = torch.randn(3, in_channels, 28, 28)
    out = mdl(x)
    # assert out.shape == torch.Size([3, 288, 28, 28])
    print(f"conv triplet test: ", out.shape)
    return mdl


def test_inner_block(in_channels, out_channels):
    mdl = model.inner_block(in_channels=in_channels, out_channels_indiv=out_channels)
    x = torch.randn(3, in_channels, 28, 28)
    out = mdl(x)
    # assert out.shape == torch.Size([3, 52, 28, 28])
    print(f"inner block: ", out.shape)
    return mdl

def test_openpose(in_channels):
    mdl = model.openpose(in_channels=in_channels)
    x = torch.randn(3, 3, 224, 224)
    out = mdl(x)
    # assert out.shape == torch.Size([3, 78, 28, 28])
    print(f"openpose: ", out[0].shape)
    return mdl


# test_vgg()
# test_conv_triplet1(in_channels=128)
# test_inner_block(in_channels=180, out_channels=52)
openpose = test_openpose(in_channels=3)
# print(openpose)