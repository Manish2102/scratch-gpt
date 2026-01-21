class TrainingSequenceService:
    '''
    Docstring for TrainingSequenceService
    '''
    def __init__(self, token_id, block_size):
        self.token_ids = token_id
        self.block_size = block_size
    
    def __len__(self):
        """
        Number of available training sequences
        """
        return len(self.token_ids) - self.block_size
    def get_sequence(self, index):
        """
        Returns ONE training pair (input, target)
        """
        if index < 0 or index >= len(self):
            raise IndexError("Training sequence index out of range")

        x = self.token_ids[index : index + self.block_size]
        y = self.token_ids[index + 1 : index + self.block_size + 1]

        return x, y