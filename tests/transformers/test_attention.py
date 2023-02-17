import pytest
import torch

from models_playground import SelfAttention


@pytest.fixture
def self_attention_module():
    return SelfAttention(embedding_dim=8, num_heads=2)


# Test SelfAttention output shape
def test_self_attention_shape():
    b = 2
    t = 4
    k = 8
    r = 2
    x = torch.randn(b, t, k)
    self_att = SelfAttention(k, r)
    y = self_att(x)
    assert y.shape == (b, t, k)
