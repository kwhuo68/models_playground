from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, reduce

from .attention import Attention, LinearAttention
from .convnext import ConvNextBlock
from .resnet import ResNetBlock


def default(val, d):
    if val is not None:
        return val
    return d


def Upsample(dim):
    """
    Add an upsampling layer

    :param dim: input dimension
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    """
    Add a downsampling layer

    :param dim: input dimension
    """
    return nn.Conv2d(dim, dim, 4, 2, 1)


class PreNorm(nn.Module):
    """PreNorm layer"""

    def __init__(self, dim, fn):
        """
        Layer of normalization before another function (i.e attention)

        :param dim: input dimension
        :param fn: function to apply
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        """
        Apply GroupNorm

        :param x: input tensor
        """
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    """Residual"""

    def __init__(self, fn):
        """
        Residual connection

        :param fn: function to apply
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Add x to the output of the function

        :param x: input tensor
        """
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings"""

    def __init__(self, dim):
        """
        Sinusoidal position embeddings

        :param dim: embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Add sinusoidal position embeddings

        :param time: input tensor of shape (b, t)
        """
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class UNet(nn.Module):
    """UNet"""

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=False,
        resnet_block_groups=4,
        use_convnext=True,
        convnext_mult=2,
    ):
        """
        UNet

        :param dim: input dimension
        :param init_dim: initial dimension
        :param out_dim: output dimension
        :param dim_mults: dimension multipliers
        :param channels: number of channels
        :param with_time_emb: whether to use time embeddings
        :param resnet_block_groups: # groups for group normalization for ResNet
        :param use_convnext: whether to use ConvNeXt blocks
        :param convnext_mult: multiplier for the number of channels in ConvNeXt blocks
        """
        super().__init__()

        # Determine dimensions
        self.channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResNetBlock, groups=resnet_block_groups)

        # Time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        """
        Apply UNet

        :param: x: input tensor
        :param: time: time tensor
        """
        x = self.init_conv(x)
        t = self.time_mlp(time) if self.time_mlp is not None else None
        h = []

        # Downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
