from ..tensor import Tensor
from . import Module

struct Linear(Module):
    var weight: Tensor
    var bias: Tensor

    fn __init__(inout self, weight: Tensor, bias: Tensor):
        self.weight = weight
        self.bias = bias

    fn __copyinit__(inout self, other: Linear):
        self.weight = other.weight
        self.bias = other.bias

    fn __moveinit__(inout self, other: Linear):
        self.weight = other.weight
        self.bias = other.bias

    fn forward(self, x: Tensor) raises -> Tensor:
        var w = self.weight
        var x_matmul = x.matmul(w.transpose(0, 1))
        # Add bias (broadcasted)
        return x_matmul.add(self.bias)
