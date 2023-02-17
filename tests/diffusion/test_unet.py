import torch
import torch.nn as nn

from models_playground import UNet


# Test UNet output shape
def test_unet_shape():
    dim = 64
    channels = 3
    batch_size = 4
    height = 128
    width = 128
    x = torch.randn(batch_size, channels, height, width)
    model = UNet(dim=dim, channels=channels)
    y = model(x, time=1)
    assert y.shape == (batch_size, channels, height, width)


# Test UNet with ConvNeXt output shape
def test_unet_convnext_shape():
    dim = 64
    channels = 3
    batch_size = 4
    height = 128
    width = 128
    x = torch.randn(batch_size, channels, height, width)
    model = UNet(dim=dim, channels=channels, use_convnext=True)
    y = model(x, time=1)
    assert y.shape == (batch_size, channels, height, width)
