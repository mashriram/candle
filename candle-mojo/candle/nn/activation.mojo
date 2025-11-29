from ..tensor import Tensor
from ..op_enums import UnaryOp
from . import Module
import math

struct Activation(Module):
    var op: Int

    alias Relu = 0
    alias Gelu = 1
    alias Sigmoid = 2

    fn __init__(inout self, op: Int):
        self.op = op

    fn forward(self, x: Tensor) raises -> Tensor:
        if self.op == 0:
            return x.relu()
        elif self.op == 1:
            return x.gelu()
        elif self.op == 2:
            return x.sigmoid() # Needs impl
        return x

    @staticmethod
    fn relu() -> Activation:
        return Activation(0)

    @staticmethod
    fn gelu() -> Activation:
        return Activation(1)

fn softmax(x: Tensor, dim: Int) raises -> Tensor:
    var max_x = x.max(dim) # We need max reduction with keepdim=True or broadcast back
    # My max(dim) reduces rank.
    # We need to reshape max_x to broadcast.
    # ...
    # Simplified: x.exp() / x.exp().sum(dim)
    var exps = x.exp()
    var sum_exps = exps.sum(dim)
    # Reshape sum_exps to broadcast back
    # (B, C) -> sum(1) -> (B,) -> (B, 1)
    # We need general unsqueeze.
    # For now assuming 2D and dim=1
    var s_shape = sum_exps.shape().dims
    var new_dims = List[Int]()
    new_dims.append(s_shape[0])
    new_dims.append(1)
    var sum_r = sum_exps.reshape(Shape(new_dims))

    return exps.div(sum_r)
