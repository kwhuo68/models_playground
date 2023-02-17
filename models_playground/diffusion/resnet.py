import torch
import torch.nn as nn
from einops import rearrange, reduce


class Block(nn.Module):
    """Base block"""

    def __init__(self, dim, dim_out, groups):
        """
        Block with convolution, normalization and activation layers

        :param dim: input dimension
        :param dim_out: output dimension
        :param groups: number of groups for group normalization
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """
        Apply convolution, normalization and activation layers

        :param x: input tensor
        :param scale_shift: optional scale and shift params
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """ResNet"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        """
        ResNet

        :param dim: input dimension
        :param dim_out: output dimension
        :param time_emb_dim: time embedding dimension
        :param groups: number of groups for group normalization
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Apply ResNet

        :param: x: input tensor
        :param: time_emb: time embedding
        """
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
