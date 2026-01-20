# Tokenization converts text into a sequence of integers called tokens and can convert them back to text.


class CharTokenizer:

    #constructor
    def __init__(self, text):

        self.text = text
        self.tokens = []
        self.tokenize()

        unique_chars = sorted(set(text))

        self.token_to_id = {}
        self.id_to_token = {}

        for index, char in enumerate(unique_chars):
            self.token_to_id[char] = index
            self.id_to_token[index] = char

    
    def encode(self, text):
        """
        Converts text into a list of token IDs
        """
        tokens = []

        for char in text:
            token_id = self.token_to_id[char]
            tokens.append(token_id)

        return tokens
    

    def decode(self, token_ids):
        """
        Converts token IDs back into text
        """
        text = ""

        for token_id in token_ids:
            text += self.id_to_token[token_id]

        return text


    #function to tokenize text into characters
    def tokenize(self):

        # Implement your tokenization logic here
        # For example, split the text into individual characters
        self.tokens = list(self.text)
    
    def get_tokens(self):
        return self.tokens