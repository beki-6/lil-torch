# lil_torch

A minimal PyTorch-inspired deep learning library built in Python. I wanted to implement the main components of PyTorch to better appreciate and understand how they work. The APIs look like PyTorch's as well. So for example, the lil_torch analog of torch.autograd.Function is lil_torch.autograd.Function.

## What’s implemented so far

- `Tensor` class with NumPy backend including major mathematical operation including +, -, \*, /, @ (Matrix Multiplication),
  power (reciprocal with other operations) and tanh (Hyberbolic Tangent for use as a non-linearity later).
- Autograd engine via `lil_torch.autograd.Function`
- Optimizer abstraction and `SGD` optimizer using the strategy pattern
- Neural Networks APIs in `lil_torch.nn`:
  - `nn.Module` base class that hijacks parameters (`nn.Parameter`) and modules (`nn.Module`) and stores them in a standard format. All other layers that have parameters or child modules will subclass it.
  - Linear Layer (`nn.Linear`)
  - Tanh Layer (`nn.Tanh`) only activiation layer for now. Might add Sigmoid or ReLu later.
  - Sequential Layer (`nn.Sequential`)
  - CrossEntropy Loss (`nn.CrossEntropyLoss`) only loss function implemented for now. might add MSELoss later.
  - Layer Normalization Layer (`nn.LayerNorm`)

## Project structure

- `lil_torch/tensor.py` — `Tensor` class and tensor utilities
- `lil_torch/autograd/function.py` — base autograd `Function` and `Context` classes
- `lil_torch/autograd/ops_builtins.py` — built-in differentiable operations including basic math operations, loss functions and layer normalization which includes mean and variance calculations.
- `lil_torch/optim/optimizer.py` — optimizer base class and parameter handling
- `lil_torch/optim/optimizers.py` — `SGD` optimizer implementation
- `lil_torch/nn/module.py` — `nn.Module` class implementation
- `lil_torch/nn/linear.py` — `nn.Linear` layer implementation
- `lil_torch/nn/tanh.py` — `nn.Tanh` layer implementation
- `lil_torch/nn/sequential.py` — `nn.Sequential` layer implementation
- `lil_torch/nn/losses.py` — contains `nn.CrossEntropyLoss` implementation
- `lil_torch/nn/layer_norm.py` — `nn.LayerNorm` implementation
- `main.py` — example usage or entry point for experiments

## Key concepts

### Tensor

The `Tensor` class wraps NumPy arrays and tracks the computational graph needed for backpropagation.

### Autograd

The autograd engine is based on `lil_torch.autograd.Function`, which defines the forward and backward behavior of differentiable operations. This enables automatic gradient computation for custom operations.

### Optimizer

`Optimizer` is designed as an extensible abstraction. The current implementation includes `SGD`, which updates parameters using gradient descent.

### nn.Module

`nn.Module` is the parent class for layers of a neural network. The current implementations include `nn.Linear`, `nn.LayerNorm` and others, which transform the input data in their own special ways be it linear transformation, activiation, loss calculation, variance normalization or other.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run examples or tests from `main.py`.

## Example usage

```python
from lil_torch.tensor import Tensor
from lil_torch.optim.optimizers import SGD

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()

print(x.grad)

optimizer = SGD([x], lr=0.01)
optimizer.step()
```

## Roadmap

Remaining pieces planned for future implementation:

- DataLoader and training utilities
- More optimizer algorithms (like Adam)

## Goals

This project is intended for learning and experimentation rather than production use. It is a hands-on way to explore how PyTorch-like systems work internally and how the components connect.

---
