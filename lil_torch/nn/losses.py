from lil_torch.autograd.ops_builtins import CrossEntropyNode
from lil_torch.autograd.function import Context
from lil_torch import Tensor

class CrossEntropyLoss():
    def __call__(self, logits, targets):
        ctx = Context()
        # call forward()
        out_data = CrossEntropyNode.forward(ctx, logits._data, targets._data.astype(int))
        out_tensor = Tensor(out_data)

        if logits.requires_grad:
            out_tensor.requires_grad = True
            out_tensor.grad_fn = (CrossEntropyNode, ctx, [logits, targets])

        return out_tensor