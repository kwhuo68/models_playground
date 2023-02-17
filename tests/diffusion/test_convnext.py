import torch

from models_playground import ConvNextBlock


# Test ConvNeXt output shape
def test_convnext_shape():
    dim = 16
    dim_out = 32
    time_emb_dim = 8
    x = torch.randn(1, dim, 28, 28)
    convnext_block = ConvNextBlock(dim=dim, dim_out=dim_out, time_emb_dim=time_emb_dim)
    y = convnext_block(x)
    assert y.shape == (1, dim_out, 28, 28)


# Test ConvNeXt with time embeddings output shape
def test_resnet_with_time_shape():
    dim = 16
    dim_out = 32
    time_emb_dim = 8
    x = torch.randn(1, dim, 28, 28)
    time_emb = torch.randn(1, time_emb_dim)
    convnext_block = ConvNextBlock(dim=dim, dim_out=dim_out, time_emb_dim=time_emb_dim)
    y = convnext_block(x, time_emb)
    assert y.shape == (1, dim_out, 28, 28)
