
class TokenizerService:
    '''
    Docstring for TokenizerService
    '''
    def __init__(self, language='en'):
        self.language = language    
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
    
    def tokenize_file(self, file_path):
        """
        Read a text file and tokenize its contents
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        return self.tokenize_to_characters(text)
    

    