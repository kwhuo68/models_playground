import torch
import torch.nn as nn
from einops import rearrange, reduce


class ConvNextBlock(nn.Module):
    """ConvNeXt"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        """
        ConvNeXt

        :param dim: input dimension
        :param dim_out: output dimension
        :param time_emb_dim: time embedding dimension
        :param mult: multiplier for the number of channels
        :param norm: whether to use normalization
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if time_emb_dim is not None
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Apply blocks

        :param: x: input tensor
        :param: time_emb: time embedding
        """
        h = self.ds_conv(x)

        if self.mlp is not None and time_emb is not None:
            assert time_emb is not None, "need time embedding"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)
