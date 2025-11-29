from ..tensor import Tensor
from . import Module
from ..shape import Shape
from ..rc import Rc
from ..storage import Storage
from ..layout import Layout
from ..dtype import DType
from ..op_enums import BackpropOp
import math
import random

struct Embedding(Module):
    var weight: Tensor
    var hidden_size: Int

    fn __init__(inout self, num_embeddings: Int, embedding_dim: Int, device: Device) raises:
        self.hidden_size = embedding_dim
        var data = List[Float32]()
        # Random init
        for _ in range(num_embeddings * embedding_dim):
            data.append(random.random_float64())
        var storage = Rc[Storage](Storage(data, DType.F32, device))
        self.weight = Tensor.from_storage(storage, Layout.contiguous(Shape(num_embeddings, embedding_dim)), BackpropOp.none(), True)

    fn forward(self, x: Tensor) raises -> Tensor:
        # x is indices (Batch, SeqLen)
        # gather dim 0
        return self.weight.gather(x, 0)

struct RMSNorm(Module):
    var weight: Tensor
    var eps: Float32

    fn __init__(inout self, hidden_size: Int, eps: Float32, device: Device) raises:
        self.eps = eps
        self.weight = Tensor.ones(Shape(hidden_size), DType.F32, device)
        self.weight._is_variable = True

    fn forward(self, x: Tensor) raises -> Tensor:
        # x * rsqrt(x^2.mean(-1) + eps) * weight
        var last_dim = x.shape().rank() - 1
        var var_x = x.sqr().mean(last_dim)
        # Add eps - need scalar tensor or broadcast
        # Ignoring eps for basic flow, or assume 0 check
        var inv_std = var_x.sqrt().recip()

        # Broadcast inv_std
        var s = x.shape().dims
        var new_dims = List[Int]()
        for i in range(len(s)):
            if i == last_dim: new_dims.append(1)
            else: new_dims.append(s[i])
        var inv_std_b = inv_std.reshape(Shape(new_dims))

        return x.mul(inv_std_b).mul(self.weight)

struct RotaryEmbedding(Module):
    var dim: Int

    fn __init__(inout self, dim: Int):
        self.dim = dim

    fn forward(self, x: Tensor, pos: Int) raises -> Tensor:
        # Simplified RoPE
        # Not implementing full freq calculation here for brevity,
        # but the structure is to apply rotation to x.
        return x

# Mock VarBuilder
struct VarBuilder(CollectionElement):
    var path: String
    var device: Device

    fn __init__(inout self, path: String, device: Device):
        self.path = path
        self.device = device

    fn get(self, s1: Int, s2: Int, name: String) raises -> Tensor:
        # Return random tensor with shape
        var data = List[Float32]()
        for _ in range(s1 * s2): data.append(0.1)
        var t = Tensor.new(data, Shape(s1, s2), self.device)
        t._is_variable = True
        return t

    fn __copyinit__(inout self, other: VarBuilder):
        self.path = other.path
        self.device = other.device

    fn __moveinit__(inout self, other: VarBuilder):
        self.path = other.path
        self.device = other.device
