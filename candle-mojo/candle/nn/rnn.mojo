from ..tensor import Tensor
from . import Module
from ..nn.linear import Linear
from ..nn.activation import Activation
from collections import List

struct RNN(Module):
    # Stub for RNN
    fn __init__(inout self):
        pass
    fn forward(self, x: Tensor) raises -> Tensor:
        return x

struct LSTM(Module):
    # Stub for LSTM
    fn __init__(inout self):
        pass
    fn forward(self, x: Tensor) raises -> Tensor:
        return x

struct GRU(Module):
    # Stub for GRU
    fn __init__(inout self):
        pass
    fn forward(self, x: Tensor) raises -> Tensor:
        return x
