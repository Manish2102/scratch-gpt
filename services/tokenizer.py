
import json


class TokenizerService:
    '''
    Docstring for TokenizerService
    '''

    # constructor for TokenizerService
    def __init__(self, vocab_path, language='en'):
        self.language = language    

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.char_to_id = json.load(f)

        # Build reverse mapping (id -> char)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        self.vocab_size = len(self.char_to_id)

    def _normalize_text(self, text):
    # Normalize Windows line endings
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')

        # Normalize whitespace
        text = text.replace('\t', ' ')
        text = text.replace('\u00a0', ' ')  # non-breaking space

        # Remove BOM if present
        text = text.replace('\ufeff', '')

        return text

    def tokenize_text(self, text):
        """
        Tokenize a string of text
        """
        if self.language == 'en':
            return text.split()

        raise NotImplementedError(
            f"Tokenization for language '{self.language}' is not implemented."
        )
    def tokenize_to_characters(self, text):
        """
        Tokenize text into individual characters
        """
        return list(text)
    
    def encode_characters(self, characters):
        """
        Encode characters into token IDs using vocabulary
        """
        token_ids = []

        for ch in characters:
            if ch not in self.char_to_id:
                raise ValueError(f"Character {repr(ch)} not in vocabulary")

            token_ids.append(self.char_to_id[ch])

        return token_ids

    
    
    def tokenize_file_to_word(self, file_path):
        """
        Read a text file and tokenize its contents
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        return self.tokenize_text(text)
    
    def tokenize_character_to_id(self, file_path):
        """
        Read a text file and tokenize its contents
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        new_text = self._normalize_text(text)
 
        characters = self.tokenize_to_characters(new_text)
        return self.encode_characters(characters)
    

    
 

    