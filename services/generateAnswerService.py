import torch
import torch.nn.functional as F


class GenerateAnswerService:
    """
    Handles text generation (inference) for GPT model
    """

    def __init__(self, model, tokenizer, temperature=1.0):
        """
        model      : trained GPTModel
        tokenizer  : TokenizerService
        temperature: controls randomness
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

        self.model.eval()

    def generate(self, prompt, max_new_tokens=200):
        """
        Generate text from a prompt
        """

        # Encode prompt → token IDs
        token_ids = []
        for ch in prompt:
            if ch not in self.tokenizer.char_to_id:
                raise ValueError(f"Character {repr(ch)} not in vocabulary")
            token_ids.append(self.tokenizer.char_to_id[ch])

        tokens = torch.tensor(token_ids, dtype=torch.long)

        # Generate tokens one by one
        for _ in range(max_new_tokens):

            # Crop context to block_size
            tokens_condensed = tokens[-self.model.block_size :]

            with torch.no_grad():
                logits, _ = self.model(tokens_condensed)

            # Take logits of last token
            logits = logits[-1] / self.temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token])

        # Decode tokens → text
        output_text = "".join(
            self.tokenizer.id_to_char[token.item()] for token in tokens
        )

        return output_text
