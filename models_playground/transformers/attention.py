import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    """Self-Attention"""

    def __init__(self, embedding_dim, num_heads, mask=False):
        """
        Self Attention

        :param embedding_dim: embedding dimension of input (vector is size k)
        :param num_heads: number of heads, i.e. r
        :param mask: if True, mask the future tokens
        """
        super().__init__()

        # Main parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mask = mask

        assert (
            embedding_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.head_dim = embedding_dim // num_heads

        # Each matrix is of size (k, k)
        self.k_w = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.q_w = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_w = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        """
        Apply self attention

        :param x: input tensor of shape (b, t, k)
        """
        b, t, k = x.size()
        h = self.num_heads
        d = k // h

        keys = self.k_w(x).view(b, t, h, d)
        queries = self.q_w(x).view(b, t, h, d)
        values = self.v_w(x).view(b, t, h, d)

        # Compute self attention

        # Fold heads into batch dimension - K, Q, V
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, d)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, d)
        values = values.transpose(1, 2).contiguous().view(b * h, t, d)

        queries = queries / (d ** (1 / 4))
        keys = keys / (d ** (1 / 4))

        # Vector setup: q_i = Qx_i, k_i = Kx_i, v_i = Vx_i

        # Compute dot product of scaled Q and K - each w'_ij = q_i * k_j
        dot_product = torch.bmm(queries, keys.transpose(1, 2))

        assert dot_product.size() == (b * h, t, t)

        # Add mask
        if self.mask:
            mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
            dot_product = dot_product.masked_fill(mask, float("-inf"))

        # Computing weights, w_ij = softmax(w'_ij)
        dot_product = F.softmax(dot_product, dim=2)

        # Compute self attention to V, i.e. y_i = sum_j(w_ij * v_j)
        s_att = torch.bmm(dot_product, values).view(b, h, t, d)

        # Unfold heads
        s_att = s_att.transpose(1, 2).contiguous().view(b, t, k)

        return s_att
