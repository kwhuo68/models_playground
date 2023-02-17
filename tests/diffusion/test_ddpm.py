import pytest
import torch

from models_playground import DDPM


@pytest.fixture
def model():
    beta_start = 0.01
    beta_end = 0.1
    timesteps = 1000
    return DDPM(beta_start, beta_end, timesteps)


# Test q_sample output shape
def test_q_sample(model):
    x_0 = torch.randn(16, 3, 32, 32)
    t = torch.zeros(16, dtype=torch.long)
    noise = torch.randn_like(x_0)
    output = model.q_sample(x_0, t, noise)
    assert output.shape == x_0.shape


# Test p_sample output shape
def test_p_sample(model):
    x = torch.randn(16, 3, 32, 32)
    t = torch.zeros(16, dtype=torch.long)
    t_index = 0
    output = model.p_sample(lambda x, t: x, x, t, t_index)
    assert output.shape == x.shape


# Test that p_losses returns a non-negative loss
def test_p_losses(model):
    x_0 = torch.randn(16, 3, 32, 32)
    t = torch.zeros(16, dtype=torch.long)
    noise = torch.randn_like(x_0)
    loss = model.p_losses(lambda x, t: x, x_0, t, noise)
    assert loss >= 0.0
