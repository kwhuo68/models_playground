import pytest
import torch

from models_playground import Attention, LinearAttention


@pytest.fixture
def input_tensor():
    return torch.randn(2, 16, 16, 16)


# Get basic attention module
@pytest.fixture(params=[Attention, LinearAttention])
def attention_module(request):
    module_cls = request.param
    return module_cls(16)


# Test Attention output shape
def test_attention_forward(attention_module, input_tensor):
    output = attention_module(input_tensor)
    assert output.shape == input_tensor.shape
