import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (GPT-style)
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),                     # GPT uses GELU
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        """
        x shape: (block_size, embedding_dim)
        return: (block_size, embedding_dim)
        """
        return self.net(x)
