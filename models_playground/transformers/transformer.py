import torch
import torch.nn.functional as F
from torch import nn

from .attention import SelfAttention


class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        seq_length,
        mask,
        ff_hidden_mult=4,
        dropout=0,
        attention_type="default",
        pos_embedding=None,
    ):
        """
        Transformer block

        :param embedding_dim: embedding dimension of input (vector is size k)
        :param num_heads: number of heads, i.e. r
        :param seq_length: length of input sequence
        :param mask: if True, mask the future tokens
        :param ff_hidden_mult: hidden dimension multiplier for feedforward layer
        :param dropout: dropout rate
        :param attention_type: type of attention to use
        :param pos_embedding: positional embedding to use
        """
        super().__init__()

        self.attention = SelfAttention(embedding_dim, num_heads, mask)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_mult * embedding_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embedding_dim, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply transformer block

        :param x: input tensor of shape (b, t, k)
        """
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)
        feedforward = self.feedforward(x)
        x = self.norm2(feedforward + x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    """Transformer"""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        depth,
        seq_length,
        num_tokens,
        attention_type="default",
        max_pool=True,
        dropout=0.0,
    ):
        """
        Transformer

        :param embedding_dim: embedding dimension of input (vector is size k)
        :param num_heads: number of heads, i.e. r
        :param depth: number of transformer blocks
        :param seq_length: length of input sequence
        :param num_tokens: number of tokens in vocabulary
        :param attention_type: type of attention to use
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.max_pool = max_pool
        self.token_embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=seq_length, embedding_dim=embedding_dim
        )

        transformer_blocks = []
        for i in range(depth):
            transformer_blocks.append(
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    seq_length=seq_length,
                    mask=False,
                    attention_type=attention_type,
                    pos_embedding=self.pos_embedding,
                )
            )

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.toprobs = nn.Linear(embedding_dim, num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply transformer

        :param x: input of size (batch, seq_length) tensor of token indices
        :return predicted log-prob vectors for each token based on prev tokens
        """
        tokens = self.token_embedding(x)
        b, t, k = tokens.size()
        positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, k)
        x = tokens + positions
        x = self.dropout(x)
        x = self.transformer_blocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over time

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)
