import numpy as np
from .module import Parameter, Module

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()