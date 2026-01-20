from services.tokenizer import TokenizerService

def main():
    '''
    Docstring for main
    '''

    tokenizer = TokenizerService(language='en')

    tokens = tokenizer.tokenize_file("datasets/big.txt")

    print("Tokens:")
    print(tokens)

if __name__ == "__main__":
    print("__name__ == __main__")
    main()
