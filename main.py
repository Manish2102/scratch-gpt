from services.tokenizer import TokenizerService
from services.trainingSequenceService import TrainingSequenceService
from services.tensorsService import TensorService
from models.embeddingModel import TokenEmbeddingModel
from models.positionalEmbeddingModel import PositionalEmbeddingModel

def main():
    """
    Docstring for main
    """
    tokenizer = TokenizerService("services/vocabulary.json", language='en')

    # Tokenize full dataset (OK)
    tokens = tokenizer.tokenize_character_to_id("datasets/big.txt")
    print("length of tokens:", len(tokens))

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

if __name__ == "__main__":
    print("__name__ == __main__")
    main()
