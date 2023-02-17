import torch

from models_playground import Transformer, TransformerBlock


# Test TransformerBlock output shape
def test_transformer_block():
    batch_size = 2
    seq_length = 4
    embedding_dim = 8
    num_heads = 2
    x = torch.randn((batch_size, seq_length, embedding_dim))
    block = TransformerBlock(embedding_dim, num_heads, seq_length, mask=False)
    output = block(x)
    assert output.shape == (batch_size, seq_length, embedding_dim)


# Test Transformer output shape
def test_transformer():
    batch_size = 2
    seq_length = 4
    embedding_dim = 8
    num_heads = 2
    depth = 2
    num_tokens = 16
    x = torch.randint(0, num_tokens, (batch_size, seq_length))
    transformer = Transformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        depth=depth,
        seq_length=seq_length,
        num_tokens=num_tokens,
        attention_type="default",
        max_pool=True,
        dropout=0.0,
    )
    output = transformer(x)
    assert output.shape == (batch_size, num_tokens)
