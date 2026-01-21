import torch.nn as nn

class TokenEmbeddingModel(nn.Module):
    """
    Docstring for TokenEmbeddingModel
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        x must be a torch.Tensor of dtype long
        shape: (block_size,)
        """
        return self.embeddings(x)
