from ..tensor import Tensor
from . import Module
from ..shape import Shape
import math

struct LayerNorm(Module):
    var weight: Tensor
    var bias: Tensor
    var eps: Float32

    fn __init__(inout self, weight: Tensor, bias: Tensor, eps: Float32):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    fn forward(self, x: Tensor) raises -> Tensor:
        # Mean and Var over last dim
        var last_dim = x.shape().rank() - 1
        var u = x.mean(last_dim)

        # Reshape u to broadcast
        var s = x.shape().dims
        var new_dims = List[Int]()
        for i in range(len(s)):
            if i == last_dim: new_dims.append(1)
            else: new_dims.append(s[i])
        var u_b = u.reshape(Shape(new_dims))

        var diff = x.sub(u_b)
        var sqr = diff.sqr()
        var var_val = sqr.mean(last_dim)
        var var_b = var_val.reshape(Shape(new_dims))

        # std = sqrt(var + eps)
        # We need scalar add.
        # eps tensor
        # ...
        var inv_std = var_b.sqrt().recip() # Ignoring eps for now or add scalar support

        var norm = diff.mul(inv_std)
        return norm.mul(self.weight).add(self.bias)
