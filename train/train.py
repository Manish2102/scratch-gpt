import torch
import torch.nn as nn
from torch.optim import AdamW

from services.tokenizer import TokenizerService
from services.trainingSequenceService import TrainingSequenceService
from services.tensorsService import TensorService
from models.getBlock import GPTBlock


class GPTModel(nn.Module):
    """Complete GPT model with embedding and output layers"""
    def __init__(self, vocab_size, block_size, embedding_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gpt_block = GPTBlock(embedding_dim=embedding_dim, block_size=block_size)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        # x shape: (block_size,)
        x = self.embedding(x)  # (block_size, embedding_dim)
        x = self.gpt_block(x)  # (block_size, embedding_dim)
        logits = self.lm_head(x)  # (block_size, vocab_size)
        return logits


def train():
    """Train the GPT model"""

    block_size = 8
    embedding_dim = 64
    num_layers = 2
    learning_rate = 3e-4
    max_steps = 10000


    tokenizer = TokenizerService("services/vocabulary.json")

    token_ids = tokenizer.tokenize_character_to_id("datasets/big.txt")
    print("Total tokens:", len(token_ids))

    dataset = TrainingSequenceService(
        token_id=token_ids,
        block_size=block_size
    )

    tensor_service = TensorService()

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )


    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()


    for step in range(max_steps):
        # Get training sequence
        x, y = dataset.get_sequence(step)

        # Convert to tensors
        x, y = tensor_service.convert_to_tensor(x, y)

        # Forward pass
        logits = model(x)  # shape: (block_size, vocab_size)

        # Calculate loss
        loss = loss_fn(logits, y)  # y shape: (block_size,)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "gpt_model.pth")

if __name__ == "__main__":
    train()
