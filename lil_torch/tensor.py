import numpy as np
from lil_torch.autograd.ops_builtins import Add, Mul, MatMul, Neg, Reciprocal, Tanh
from lil_torch.autograd.function import Context

# uses NumPy as a backend 
class Tensor:
    # initialization 
    # data- from another tensor, from numpy or from a python list
    # requires_grad (boolean, false by default)
    # class factory pattern to emulate PyTorch's torch.tensor() - requires having FloatTensor and IntTensor classes
    def __init__(self, data, requires_grad=False):
        if isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = np.array(data, dtype=np.float64)
        
        # initializing grad and grad_fn
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    # dynamic attribute for shape (tensor.shape)
    @property
    def shape(self):
        return self._data.shape
    
    # data type (tensor.dtype)
    @property
    def dtype(self):
        return self._data.dtype
    
    # number of dimensions (tensor.ndim)
    @property
    def ndim(self):
        return self._data.ndim

    # len(tensor)
    def __len__(self):
        return len(self._data)

    # indexing, slicing 
    def __getitem__(self, indices):
        res = self._data[indices]
        return Tensor(res)
    
    def __setitem__(self, indices, value):
        raw_val = value._data if isinstance(value, Tensor) else value
        self._data[indices] = raw_val

    # truthiness (bool(tensor))

    # mathematical operations -> if self.requires_grad is True, create a node in a DAG

    # add - elementwise (+)
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        ctx = Context()
        out_data = Add.forward(ctx, self._data, other._data)
        out_tensor = Tensor(out_data)

        if self.requires_grad or other.requires_grad:
            out_tensor.requires_grad=True
            out_tensor.grad_fn = (Add, ctx, [self, other])

        return out_tensor

    def __radd__(self, other):
        return self.__add__(other)
    
    # negation (-Tensor)
    def __neg__(self):
        ctx = Context()
        out_data = Neg.forward(ctx, self._data)
        out_tensor = Tensor(out_data)

        if self.requires_grad:
            out_tensor.requires_grad=True
            out_tensor.grad_fn = (Neg, ctx, [self,])

        return out_tensor
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # multiply - elementwise (*)
    def __mul__(self, other):
        # wrap other in Tensor if it's not one 
        other = other if isinstance(other, Tensor) else Tensor(other)

        ctx = Context()
        out_data = Mul.forward(ctx, self._data, other._data)
        out_tensor = Tensor(out_data)

        if self.requires_grad or other.requires_grad:
            out_tensor.requires_grad=True
            out_tensor.grad_fn = (Mul, ctx, [self, other])

        return out_tensor

    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    # multiply - elementwise (*)
    def __truediv__(self, other):
        # wrap other in Tensor if it's not one 
        other = other if isinstance(other, Tensor) else Tensor(other)

        ctx = Context()
        out_data = Reciprocal.forward(ctx, self._data)
        out_tensor = Tensor(out_data)

        if self.requires_grad:
            out_tensor.requires_grad=True
            out_tensor.grad_fn = (Reciprocal, ctx, [self,])

        return self * out_tensor
    
    # matrix multiply (@)
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        ctx = Context()
        out_data = MatMul.forward(ctx, self._data, other._data)
        out_tensor = Tensor(out_data)

        if self.requires_grad or other.requires_grad:
            out_tensor.requires_grad=True
            out_tensor.grad_fn = (MatMul, ctx, [self, other])

        return out_tensor

    # string representation
    def __repr__(self):
        return f"Tensor(data={self._data.tolist()}, dtype={self.dtype})"
    
    # Non-linearities and other mathematical functions

    # Tanh non-linearity
    def tanh(self):
        ctx = Context()
        out_data = Tanh.forward(ctx, self._data)
        out_tensor = Tensor(out_data)
        
        if self.requires_grad:
            out_tensor.requires_grad = True
            out_tensor.grad_fn = (Tanh, ctx, [self])
            
        return out_tensor
    
    # backward() method
    def backward(self, external_grad=None):
        # if requires_grad is False, return 
        if not self.requires_grad:
            return 
        
        # if this is the root node (as in the scalar loss)
        if external_grad is None:
            if self._data.size == 1:
                # assign ones
                external_grad = np.ones_like(self._data) # v = 1.0
            else:
                raise RuntimeError("Implicit grad can only be assigned for scalars")

        # accumulate external grad 
        if self.grad is None:
            self.grad = external_grad
        else:
            self.grad += external_grad

        # if not a leaf node, propagate backward 
        if self.grad_fn is not None:
            func_class, ctx, parents = self.grad_fn

            # execute VJP
            parent_grads = func_class.backward(ctx, external_grad)

            # recursively call backward on the parents 
            for parent, parent_grad in zip(parents, parent_grads):
                parent.backward(parent_grad)