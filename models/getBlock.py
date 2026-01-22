import torch.nn as nn
from models.selfAttention import CausalSelfAttention
from models.feedforwardModel import FeedForwardNetwork

class GPTBlock(nn.Module):
    """
    Single GPT Transformer Block
    """

    def __init__(self, embedding_dim, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            block_size=block_size
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim)

    def forward(self, x):
        """
        x shape: (block_size, embedding_dim)
        """

        # Residual + Attention
        x = x + self.attention(self.ln1(x))

        # Residual + Feed Forward
        x = x + self.ffn(self.ln2(x))

        return x
