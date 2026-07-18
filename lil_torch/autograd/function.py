from abc import ABC, abstractmethod

class Context:
    # initializes an empty Tensor list 
    def __init__(self):
        self.saved_tensors = []
    # method for extending the Tensor list
    def save_for_backward(self, *tensor):
        self.saved_tensors.extend(tensor)
        
# abstract base class (ABC) definition for Fuction class 
class Function(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx, *args):
        '''Subclass must implement this method in its forward pass operations.'''
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, external_grad):
        '''Subclass must calculate VJP for backpropagation.'''
        pass