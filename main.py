import torch
import os
from services.tokenizer import TokenizerService
from services.trainingSequenceService import TrainingSequenceService
from services.tensorsService import TensorService
from models.embeddingModel import TokenEmbeddingModel
from models.positionalEmbeddingModel import PositionalEmbeddingModel
from models.selfAttention import CausalSelfAttention
from models.feedforwardModel import FeedForwardNetwork
from models.getBlock import GPTBlock
from services.generateAnswerService import GenerateAnswerService
from train import GPTModel

def main():
    """
    Docstring for main
    """
    tokenizer = TokenizerService("services/vocabulary.json", language='en')

    # Tokenize full dataset (OK)
    tokens = tokenizer.tokenize_character_to_id("datasets/big.txt")
    print("length of tokens:", len(tokens))

    # Load checkpoint if it exists
    model = None
    if os.path.exists("gpt_checkpoint.pth"):
        # Create model and load checkpoint
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            embedding_dim=64,
            num_layers=2
        )
        checkpoint = torch.load("gpt_model.pth")
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.")
    else:
        print("Checkpoint not found. Skipping checkpoint loading.")


    #  Create training sequence service
    sequence_service = TrainingSequenceService(tokens, block_size=8)

    # Get ONE training sequence
    x, y = sequence_service.get_sequence(0)

    # Convert to tensors (CRITICAL)
    tensor_service = TensorService()
    x_tensor, y_tensor = tensor_service.convert_to_tensor(x, y)

    print("X tensor:", x_tensor)
    print("Y tensor:", y_tensor)

    # Create embedding model
    embedder = TokenEmbeddingModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64
    )

    # 6️Apply embeddings to INPUT tensor ONLY
    embedded_x = embedder(x_tensor)
    embedder_y = embedder(y_tensor)

    print("Embedded X shape:", embedded_x.shape)
    print("Embedded X dtype:", embedded_x.dtype)

    print("Embedded Y shape:", embedder_y.shape)
    print("Embedded Y dtype:", embedder_y.dtype)

    positional_embedding = PositionalEmbeddingModel(block_size=8, embedding_dim=64)

    final_embeddings = positional_embedding(embedded_x)

    print("Final embeddings shape:", final_embeddings.shape)

    attention = CausalSelfAttention(
    embedding_dim=64,
    block_size=8
    )

    attended_output = attention(final_embeddings)

    print("Attention output shape:", attended_output.shape)

    ffn = FeedForwardNetwork(embedding_dim=64)

    ffn_output = ffn(attended_output)

    print("FFN output shape:", ffn_output.shape)


    gpt_block = GPTBlock(
        embedding_dim=64, block_size=8
    )

    gpt_block_output = gpt_block(final_embeddings)

    print("GPT block output shape:", gpt_block_output.shape)

    # Only generate if checkpoint exists
    if model is not None:
        generator = GenerateAnswerService(
            model=model,
            tokenizer=tokenizer,
            temperature=1.0
        )

        prompt = "Hello"
        output = generator.generate(prompt, max_new_tokens=300)

        print(output)
    else:
        print("No checkpoint available for text generation.")

if __name__ == "__main__":
    print("__name__ == __main__")
    main()
