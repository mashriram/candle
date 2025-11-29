from collections import List
from .dtype import DType
from .device import Device
from .shape import Shape
from .error import Error
from .layout import Layout
from .op_enums import BinaryOp as BinOpEnum
from .op_enums import UnaryOp as UnOpEnum
import math
import random

# --- Forward Declarations ---
# In Mojo, we define structs in order. `Op` needs `Tensor`, `Tensor` needs `Op`.
# Since we are using struct-based value semantics, we can't easily cycle.
# Strategy: `Op` holds `Tensor`. `Tensor` holds `BackpropOp` which holds `Op`.
# To break cycle, we used `BackpropOp` wrapping `Op`.
# But `Op` needs `Tensor`.
# We will define `Tensor` first, but `Tensor` needs `BackpropOp`.
# We need to defer the definition of `Op` content or use a simplified handle.
# Actually, since Mojo structs are values, `Tensor` cannot contain `Tensor` directly (infinite size).
# One of them MUST use a pointer/indirection.
# `List` provides indirection (it's a handle to heap).
# So `Op` holding `List[Tensor]` is fine!
# `Tensor` holding `BackpropOp` holding `Op` holding `List[Tensor]` is fine!
# Cycle is broken by `List` being a reference/handle type.

struct Op(CollectionElement):
    var tag: Int
    var lhs: List[Tensor]
    var bin_op: Int
    var un_op: Int

    fn __init__(inout self):
        self.tag = 0
        self.lhs = List[Tensor]()
        self.bin_op = 0
        self.un_op = 0

    @staticmethod
    fn binary(lhs: Tensor, rhs: Tensor, op: Int) -> Op:
        var o = Op()
        o.tag = 1
        o.lhs.append(lhs)
        o.lhs.append(rhs)
        o.bin_op = op
        return o

    @staticmethod
    fn unary(arg: Tensor, op: Int) -> Op:
        var o = Op()
        o.tag = 2
        o.lhs.append(arg)
        o.un_op = op
        return o

    @staticmethod
    fn matmul(lhs: Tensor, rhs: Tensor) -> Op:
        var o = Op()
        o.tag = 3
        o.lhs.append(lhs)
        o.lhs.append(rhs)
        return o

    fn __copyinit__(inout self, other: Op):
        self.tag = other.tag
        self.lhs = other.lhs
        self.bin_op = other.bin_op
        self.un_op = other.un_op

    fn __moveinit__(inout self, other: Op):
        self.tag = other.tag
        self.lhs = other.lhs
        self.bin_op = other.bin_op
        self.un_op = other.un_op

struct BackpropOp(CollectionElement):
    var op: Op
    var has_op: Bool

    fn __init__(inout self):
        self.op = Op()
        self.has_op = False

    fn __init__(inout self, op: Op):
        self.op = op
        self.has_op = True

    fn __copyinit__(inout self, other: BackpropOp):
        self.op = other.op
        self.has_op = other.has_op

    fn __moveinit__(inout self, other: BackpropOp):
        self.op = other.op
        self.has_op = other.has_op

    @staticmethod
    fn none() -> BackpropOp:
        return BackpropOp()

struct Storage(CollectionElement):
    var data: List[Float32]
    var dtype: DType
    var device: Device

    fn __init__(inout self, data: List[Float32], dtype: DType, device: Device):
        self.data = data
        self.dtype = dtype
        self.device = device

    fn __copyinit__(inout self, other: Storage):
        self.data = other.data
        self.dtype = other.dtype
        self.device = other.device

    fn __moveinit__(inout self, other: Storage):
        self.data = other.data
        self.dtype = other.dtype
        self.device = other.device

fn generate_id() -> Int:
    # Use random number as ID for now.
    return int(random.random_float64() * 10000000.0)

# Struct to return gradients mapping
struct Grads(CollectionElement):
    var ids: List[Int]
    var grads: List[Tensor]

    fn __init__(inout self):
        self.ids = List[Int]()
        self.grads = List[Tensor]()

    fn __copyinit__(inout self, other: Grads):
        self.ids = other.ids
        self.grads = other.grads

    fn __moveinit__(inout self, other: Grads):
        self.ids = other.ids
        self.grads = other.grads

struct Tensor(CollectionElement, Stringable, EqualityComparable):
    var _storage: Storage
    var _layout: Layout
    var _id: Int
    var _op: BackpropOp
    var _is_variable: Bool

    fn __init__(inout self, storage: Storage, layout: Layout, op: BackpropOp, is_variable: Bool):
        self._storage = storage
        self._layout = layout
        self._id = generate_id()
        self._op = op
        self._is_variable = is_variable

    fn __copyinit__(inout self, other: Tensor):
        self._storage = other._storage
        self._layout = other._layout
        self._id = other._id
        self._op = other._op
        self._is_variable = other._is_variable

    fn __moveinit__(inout self, other: Tensor):
        self._storage = other._storage
        self._layout = other._layout
        self._id = other._id
        self._op = other._op
        self._is_variable = other._is_variable

    fn __eq__(self, other: Tensor) -> Bool:
        return self._id == other._id

    fn __ne__(self, other: Tensor) -> Bool:
        return self._id != other._id

    fn shape(self) -> Shape:
        return self._layout.shape()

    fn stride(self) -> List[Int]:
        return self._layout.stride()

    fn dtype(self) -> DType:
        return self._storage.dtype

    fn device(self) -> Device:
        return self._storage.device

    fn layout(self) -> Layout:
        return self._layout

    fn id(self) -> Int:
        return self._id

    fn op(self) -> BackpropOp:
        return self._op

    fn is_variable(self) -> Bool:
        return self._is_variable

    fn __str__(self) -> String:
        return "Tensor[" + str(self.shape()) + ", " + self.dtype().as_str() + "]"

    # --- Creation methods ---

    @staticmethod
    fn from_storage(storage: Storage, layout: Layout, op: BackpropOp, is_variable: Bool) -> Tensor:
        return Tensor(storage, layout, op, is_variable)

    @staticmethod
    fn new(data: List[Float32], shape: Shape, device: Device) raises -> Tensor:
        if shape.elem_count() != len(data):
            raise Error("Shape size does not match data length")
        var storage = Storage(data, DType.F32, device)
        var layout = Layout.contiguous(shape)
        return Tensor.from_storage(storage, layout, BackpropOp.none(), False)

    @staticmethod
    fn zeros(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for _ in range(count):
            data.append(0.0)
        return Tensor.from_storage(Storage(data, dtype, device), Layout.contiguous(shape), BackpropOp.none(), False)

    @staticmethod
    fn ones(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for _ in range(count):
            data.append(1.0)
        return Tensor.from_storage(Storage(data, dtype, device), Layout.contiguous(shape), BackpropOp.none(), False)

    fn zeros_like(self) raises -> Tensor:
        return Tensor.zeros(self.shape(), self.dtype(), self.device())

    fn ones_like(self) raises -> Tensor:
        return Tensor.ones(self.shape(), self.dtype(), self.device())

    # --- Indexing and Movement ---

    fn reshape(self, shape: Shape) raises -> Tensor:
        if shape.elem_count() != self.shape().elem_count():
             raise Error("Shape mismatch in reshape")
        if not self._layout.is_contiguous():
             raise Error("Reshape currently only supports contiguous tensors")
        return Tensor.from_storage(self._storage, Layout.contiguous(shape), BackpropOp.none(), self._is_variable)

    fn transpose(self, dim1: Int, dim2: Int) raises -> Tensor:
        var new_layout = self._layout.transpose(dim1, dim2)
        return Tensor.from_storage(self._storage, new_layout, BackpropOp.none(), self._is_variable)

    fn narrow(self, dim: Int, start: Int, length: Int) raises -> Tensor:
        var new_layout = self._layout.narrow(dim, start, length)
        return Tensor.from_storage(self._storage, new_layout, BackpropOp.none(), self._is_variable)

    fn broadcast_as(self, shape: Shape) raises -> Tensor:
        var new_layout = self._layout.broadcast_as(shape)
        return Tensor.from_storage(self._storage, new_layout, BackpropOp.none(), self._is_variable)

    # --- Operations ---

    fn flatten_to_list(self) -> List[Float32]:
        if self._layout.is_contiguous():
            var start = self._layout.start_offset()
            var count = self._layout.shape().elem_count()
            var res = List[Float32](capacity=count)
            for i in range(count):
                res.append(self._storage.data[start + i])
            return res

        var shape = self.shape()
        var dims = shape.dims
        var stride = self._layout.stride()
        var offset = self._layout.start_offset()
        var count = shape.elem_count()
        var res = List[Float32](capacity=count)

        for i in range(count):
            var temp_i = i
            var current_offset = offset
            for d in range(len(dims) - 1, -1, -1):
                var coord = temp_i % dims[d]
                temp_i = temp_i // dims[d]
                current_offset += coord * stride[d]
            res.append(self._storage.data[current_offset])
        return res

    fn matmul(self, rhs: Tensor) raises -> Tensor:
        var lhs_dims = self.shape().dims
        var rhs_dims = rhs.shape().dims

        if len(lhs_dims) < 2 or len(rhs_dims) < 2:
             raise Error("Matmul requires at least 2 dimensions")

        var m = lhs_dims[len(lhs_dims) - 2]
        var k = lhs_dims[len(lhs_dims) - 1]
        var k2 = rhs_dims[len(rhs_dims) - 2]
        var n = rhs_dims[len(rhs_dims) - 1]

        if k != k2:
             raise Error("Matmul dimension mismatch")

        var lhs_data = self.flatten_to_list()
        var rhs_data = rhs.flatten_to_list()

        var res_shape = Shape(m, n)
        var res_data = List[Float32](capacity=m*n)
        for _ in range(m*n):
            res_data.append(0.0)

        for i in range(m):
            for j in range(n):
                var sum: Float32 = 0.0
                for l in range(k):
                    var val_a = lhs_data[i * k + l]
                    var val_b = rhs_data[l * n + j]
                    sum += val_a * val_b
                res_data[i * n + j] = sum

        var op = BackpropOp.none()
        if self._is_variable or rhs._is_variable or self._op.has_op or rhs._op.has_op:
            op = BackpropOp(Op.matmul(self, rhs))

        return Tensor.from_storage(Storage(res_data, self.dtype(), self.device()), Layout.contiguous(res_shape), op, False)

    fn _binary_op(self, rhs: Tensor, op_code: Int) raises -> Tensor:
        var shape = self.shape().broadcast_shape_binary_op(rhs.shape())

        var lhs_b = self.broadcast_as(shape)
        var rhs_b = rhs.broadcast_as(shape)

        var lhs_data = lhs_b.flatten_to_list()
        var rhs_data = rhs_b.flatten_to_list()
        var count = shape.elem_count()
        var res_data = List[Float32](capacity=count)

        for i in range(count):
            var a = lhs_data[i]
            var b = rhs_data[i]
            var val: Float32 = 0.0
            if op_code == BinOpEnum.Add:
                val = a + b
            elif op_code == BinOpEnum.Sub:
                val = a - b
            elif op_code == BinOpEnum.Mul:
                val = a * b
            elif op_code == BinOpEnum.Div:
                val = a / b
            res_data.append(val)

        var op = BackpropOp.none()
        if self._is_variable or rhs._is_variable or self._op.has_op or rhs._op.has_op:
            op = BackpropOp(Op.binary(self, rhs, op_code))

        return Tensor.from_storage(Storage(res_data, self.dtype(), self.device()), Layout.contiguous(shape), op, False)

    fn add(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Add)

    fn sub(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Sub)

    fn mul(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Mul)

    fn div(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Div)

    fn _unary_op(self, op_code: Int) raises -> Tensor:
        var data = self.flatten_to_list()
        var count = len(data)
        var res_data = List[Float32](capacity=count)

        for i in range(count):
            var v = data[i]
            var r: Float32 = 0.0
            if op_code == UnOpEnum.Neg:
                r = -v
            elif op_code == UnOpEnum.Exp:
                r = math.exp(v)
            elif op_code == UnOpEnum.Log:
                r = math.log(v)
            elif op_code == UnOpEnum.Sqr:
                r = v * v
            elif op_code == UnOpEnum.Sqrt:
                r = math.sqrt(v)
            elif op_code == UnOpEnum.Recip:
                r = 1.0 / v
            elif op_code == UnOpEnum.Sin:
                r = math.sin(v)
            elif op_code == UnOpEnum.Cos:
                r = math.cos(v)
            res_data.append(r)

        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            op = BackpropOp(Op.unary(self, op_code))

        return Tensor.from_storage(Storage(res_data, self.dtype(), self.device()), Layout.contiguous(self.shape()), op, False)

    fn neg(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Neg)

    fn exp(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Exp)

    fn log(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Log)

    fn sqr(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Sqr)

    fn sqrt(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Sqrt)

    fn recip(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Recip)

    fn sin(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Sin)

    fn cos(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Cos)

    # --- Backprop ---

    fn sorted_nodes(self) -> List[Tensor]:
        var nodes = List[Tensor]()
        var visited = List[Int]()
        self._walk(nodes, visited)
        return nodes

    fn _walk(self, inout nodes: List[Tensor], inout visited: List[Int]):
        for v in visited:
            if v[] == self._id:
                return

        if self._is_variable:
            pass
        elif self._op.has_op:
            var op = self._op.op
            for i in range(len(op.lhs)):
                op.lhs[i]._walk(nodes, visited)

        visited.append(self._id)
        nodes.append(self)

    fn backward(self) raises -> Grads:
        var sorted = self.sorted_nodes()
        var reverse_sorted = List[Tensor]()
        for i in range(len(sorted)-1, -1, -1):
            reverse_sorted.append(sorted[i])

        var grads = Grads()

        # Seed with 1.0 for self
        grads.ids.append(self._id)
        grads.grads.append(self.ones_like())

        for i in range(len(reverse_sorted)):
            var node = reverse_sorted[i]
            if node._is_variable:
                continue

            var grad: Tensor
            var found = False
            for k in range(len(grads.ids)):
                if grads.ids[k] == node._id:
                    grad = grads.grads[k]
                    found = True
                    break

            if not found:
                continue

            if node._op.has_op:
                var op = node._op.op
                if op.tag == 1: # Binary
                    var lhs = op.lhs[0]
                    var rhs = op.lhs[1]
                    var code = op.bin_op

                    if code == BinOpEnum.Add:
                        self._accumulate_grad(grads, lhs, grad)
                        self._accumulate_grad(grads, rhs, grad)
                    elif code == BinOpEnum.Sub:
                        self._accumulate_grad(grads, lhs, grad)
                        self._accumulate_grad(grads, rhs, grad.neg())
                    elif code == BinOpEnum.Mul:
                         self._accumulate_grad(grads, lhs, grad.mul(rhs))
                         self._accumulate_grad(grads, rhs, grad.mul(lhs))

                elif op.tag == 3: # Matmul
                    var lhs = op.lhs[0]
                    var rhs = op.lhs[1]
                    var lhs_grad = grad.matmul(rhs.transpose(0, 1))
                    var rhs_grad = lhs.transpose(0, 1).matmul(grad)

                    self._accumulate_grad(grads, lhs, lhs_grad)
                    self._accumulate_grad(grads, rhs, rhs_grad)

                elif op.tag == 2: # Unary
                    var arg = op.lhs[0]
                    var code = op.un_op
                    if code == UnOpEnum.Neg:
                        self._accumulate_grad(grads, arg, grad.neg())
                    elif code == UnOpEnum.Exp:
                        # d(e^x) = e^x * g
                        self._accumulate_grad(grads, arg, grad.mul(node)) # node is e^x
                    elif code == UnOpEnum.Sqr:
                        # d(x^2) = 2x * g
                        # 2 * arg * grad
                        var two_data = List[Float32]()
                        two_data.append(2.0)
                        var two = Tensor.new(two_data, Shape(1), arg.device())

                        var two_arg = arg.mul(two) # arg * 2 (broadcast)
                        self._accumulate_grad(grads, arg, grad.mul(two_arg))

        return grads

    fn _accumulate_grad(self, inout grads: Grads, target: Tensor, g: Tensor) raises:
        for i in range(len(grads.ids)):
            if grads.ids[i] == target._id:
                grads.grads[i] = grads.grads[i].add(g)
                return
        grads.ids.append(target._id)
        grads.grads.append(g)
