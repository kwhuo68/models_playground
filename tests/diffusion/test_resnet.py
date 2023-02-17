import torch

from models_playground import ResNetBlock


# Test ResNet output shape
def test_resnet_shape():
    dim = 16
    dim_out = 32
    time_emb_dim = 8
    groups = 8
    x = torch.randn(1, dim, 28, 28)
    resnet_block = ResNetBlock(
        dim=dim, dim_out=dim_out, time_emb_dim=time_emb_dim, groups=groups
    )
    y = resnet_block(x)
    assert y.shape == (1, dim_out, 28, 28)


# Test ResNet with time embeddings output shape
def test_resnet_with_time_shape():
    dim = 16
    dim_out = 32
    time_emb_dim = 8
    groups = 8
    x = torch.randn(1, dim, 28, 28)
    time_emb = torch.randn(1, time_emb_dim)
    resnet_block = ResNetBlock(
        dim=dim, dim_out=dim_out, time_emb_dim=time_emb_dim, groups=groups
    )
    y = resnet_block(x, time_emb)
    assert y.shape == (1, dim_out, 28, 28)
