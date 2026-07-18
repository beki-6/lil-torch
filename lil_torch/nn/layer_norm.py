import numpy as np
from lil_torch.nn.module import Parameter, Module
from lil_torch.autograd.ops_builtins import LayerNormNode
from lil_torch.autograd.function import Context
from lil_torch import Tensor 

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps 

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        # initialize gamma to 1.0 and beta to 0.0
        self.gamma = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
        self.beta = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))

    def forward(self, x):
        ctx = Context()

        # execute forward pass 
        out_data = LayerNormNode.forward(ctx, x._data, self.gamma._data, self.beta._data, self.eps)
        out_tensor = Tensor(out_data)

        if x.requires_grad:
            out_tensor.requires_grad = True
            out_tensor.grad_fn = (LayerNormNode, ctx, [x, self.gamma, self.beta])

        return out_tensor