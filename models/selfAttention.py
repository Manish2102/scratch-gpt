import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Single-head causal self-attention
    """

    def __init__(self, embedding_dim, block_size):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.block_size = block_size

        # Linear projections for Q, K, V
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key   = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Register causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """
        x shape: (block_size, embedding_dim)
        """

        B, D = x.size()  

        # 1️ Create Q, K, V
        Q = self.query(x)  # (B, D)
        K = self.key(x)    # (B, D)
        V = self.value(x)  # (B, D)

        # 2 Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        # shape: (B, B)

        # 3Apply causal mask (prevent looking ahead)
        scores = scores.masked_fill(
            self.mask[:B, :B] == 0,
            float('-inf')
        )

        # 4Softmax → attention weights
        weights = F.softmax(scores, dim=-1)

        # 5Weighted sum of values
        out = weights @ V  # (B, D)

        return out
