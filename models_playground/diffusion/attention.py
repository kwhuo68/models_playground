import torch
from einops import rearrange
from torch import einsum, nn


class Attention(nn.Module):
    """Attention"""

    def __init__(self, dim, num_heads=4, dim_head=32):
        """
        Attention

        :param dim: input dimension
        :param num_heads: number of heads
        :param dim_head: dimension of each head
        """
        super().__init__()
        self.scale = dim_head**-0.5  # sqrt(k)
        self.num_heads = num_heads
        hidden_dim = dim_head * num_heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_output = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Apply attention

        :param x: input tensor
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        output = einsum("b h i j, b h d j -> b h i d", attn, v)
        output = rearrange(output, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_output(output)


class LinearAttention(nn.Module):
    """Linear Attention"""

    def __init__(self, dim, num_heads=4, dim_head=32):
        """
        Linear Attention

        :param dim: input dimension
        :param num_heads: number of num_heads
        :param dim_head: dimension of each head
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.num_heads = num_heads
        hidden_dim = dim_head * num_heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_output = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        """
        Apply linear attention

        :param x: input tensor
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        output = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        output = rearrange(
            output, "b h c (x y) -> b (h c) x y", h=self.num_heads, x=h, y=w
        )
        return self.to_output(output)
