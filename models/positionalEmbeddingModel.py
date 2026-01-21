import torch
import torch.nn as nn

class PositionalEmbeddingModel(nn.Module):
    """
    Learnable positional embeddings
    """

    def __init__(self, block_size, embedding_dim):
        super().__init__()

        self.position_embeddings = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        """
        x: token embeddings of shape (block_size, embedding_dim)
        """
        block_size = x.size(0)

        # Create position indices: [0, 1, 2, ..., block_size-1]
        positions = torch.arange(block_size, device=x.device)

        # Get positional embeddings
        pos_emb = self.position_embeddings(positions)

        # Add token + positional embeddings
        return x + pos_emb
