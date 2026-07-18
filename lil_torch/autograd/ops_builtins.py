import numpy as np
from .function import Function

# Add Function subclass 
class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, external_grad):
        dy = external_grad
        dx = external_grad
        return dx, dy
    
# Mul Function subclass 
class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, external_grad):
        x, y = ctx.saved_tensors
        dy = external_grad * x
        dx = external_grad * y
        return dx, dy
    
# MatMul Function subclass 
class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(ctx, external_grad):
        x, y = ctx.saved_tensors
        dx = external_grad @ y.T
        dy = x.T @ external_grad
        return dx, dy
    
# Pow Function subclass 
class Reciprocal(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x)
        return 1.0 / x 

    @staticmethod
    def backward(ctx, external_grad):
        x, = ctx.saved_tensors
        dx = external_grad * (-1.0 / (x ** 2.0))
        return dx
    
# Neg Function subclass 
class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, external_grad):
        dx = -external_grad
        return dx
    
# Tanh Function subclass 
class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        y = np.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, external_grad):
        y, = ctx.saved_tensors
        dx = external_grad * (1.0 - y ** 2)
        return dx
    
# CrossEntropy Node 
class CrossEntropyNode(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        # get the shape of the logits 
        n, c = logits.shape # (N, C)
        # apply LogSumExp trick on the softmax denominator to avoid inf for larger logits during exponentiation 
        max_logit = np.max(logits, axis=1, keepdims=True) # (N, 1)
        shifted_logits = logits - max_logit # shifted_logits.shape = (N, C)
        log_sum_exp = max_logit + np.log(np.sum(shifted_logits, axis=1, keepdims=True)) # log_sum_exp.shape = (N, 1)
        # logarithm of the softmax equation (Log-softmax)
        log_probs = logits - log_sum_exp # (N, C)

        # negative log-likelihood 
        loss = -np.mean(log_probs[np.arange(n), targets]) # log_probs' shape after indexing = (N,), loss becomes a scalar

        # save probs (raw softmax values) for backward pass
        probs = np.exp(log_probs)
        ctx.save_for_backward(probs, targets)
        return loss 
    
    @staticmethod
    def backward(ctx, external_grad):
        probs, targets = ctx.saved_tensors
        n, c = probs.shape 

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), targets] = 1.0 

        dx = external_grad * (probs - one_hot) / n

        return dx, None 
    
class LayerNormNode(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-5):
        # calculate mean 
        mean = np.mean(x, axis=-1, keepdims=True) # (N,1)
        # calculate variance
        variance = np.var(x, axis=-1, keepdims=True) 
        # normalize 
        x_hat = (x - mean) / np.sqrt(variance + eps)
        # linear transformation 
        y = gamma * x_hat + beta 
        # save tensors for backward pass in the context 
        ctx.save_for_backward(x, gamma, x_hat, variance, eps)
        return y

    @staticmethod
    def backward(ctx, external_grad):
        # unpack ctx
        x, gamma, x_hat, variance, eps = ctx.saved_tensors
        v = external_grad # upstream gradient vector
        N, D = x.shape 

        # gradients for learnable parameters gamma and beta 
        dgamma = np.sum(v * x_hat, axis=0)
        dbeta = np.sum(v, axis=0)

        inv_std = 1.0 / np.sqrt(variance + eps)

        sum_v = np.sum(v * gamma, axis=-1, keepdims=True)
        sum_v_xhat = np.sum(v * gamma * x_hat, axis=-1, keepdims=True)

        dx = (gamma * inv_std / D) * (D * v - sum_v - x_hat * sum_v_xhat)

        return dx, dgamma, dbeta