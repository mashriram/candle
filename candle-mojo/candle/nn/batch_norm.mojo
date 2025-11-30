from ..tensor import Tensor
from . import Module
from ..shape import Shape
from ..dtype import DType
from ..device import Device
import math

struct BatchNorm(Module):
    var running_mean: Tensor
    var running_var: Tensor
    var weight: Tensor
    var bias: Tensor
    var eps: Float32
    var momentum: Float32
    var training: Bool

    fn __init__(inout self, num_features: Int, eps: Float32, momentum: Float32, device: Device) raises:
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.running_mean = Tensor.zeros(Shape(num_features), DType.F32, device)
        self.running_var = Tensor.ones(Shape(num_features), DType.F32, device)
        self.weight = Tensor.ones(Shape(num_features), DType.F32, device)
        self.bias = Tensor.zeros(Shape(num_features), DType.F32, device)

        # Mark params
        self.weight._is_variable = True
        self.bias._is_variable = True

    fn forward(self, x: Tensor) raises -> Tensor:
        # x: (N, C, H, W) or (N, C)
        # Norm over (N, H, W)
        # dims: 0, 2, 3...
        # Simplified for 2D (N, C)
        var mean: Tensor
        var var_val: Tensor

        if self.training:
            mean = x.mean(0) # Assuming (N, C) -> mean over N -> (C)
            # Update running
            # self.running_mean = self.running_mean * (1-m) + mean * m
            # We need no_grad block for this.

            var diff = x.sub(mean.broadcast_as(x.shape()))
            var_val = diff.sqr().mean(0)
        else:
            mean = self.running_mean
            var_val = self.running_var

        var inv_std = var_val.sqrt().recip() # + eps

        var x_norm = x.sub(mean.broadcast_as(x.shape())).mul(inv_std.broadcast_as(x.shape()))
        return x_norm.mul(self.weight.broadcast_as(x.shape())).add(self.bias.broadcast_as(x.shape()))
