import torch


class TensorService:
    '''
    Docstring for TensorService
    '''

    def convert_to_tensor(self, x, y):
        """
        Convert data to tensor format
        """
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        return x_tensor, y_tensor
    
    