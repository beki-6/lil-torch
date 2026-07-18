import numpy as np
from .module import Parameter, Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = np.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-bound, bound, (in_features, out_features)))
        if bias:
            self.bias = Parameter(np.random.uniform(-bound, bound, (out_features,)))

    def forward(self, x):
        return x @ self.weight + self.bias if self.bias else x @ self.weight