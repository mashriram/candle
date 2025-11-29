from collections import List
from .dtype import DType
from .device import Device
from .shape import Shape
from .error import Error
from .layout import Layout
from .storage import Storage
from .rc import Rc
from .op_enums import BinaryOp as BinOpEnum
from .op_enums import UnaryOp as UnOpEnum
import math
import random

fn generate_id() -> Int:
    return int(random.random_float64() * 1000000000.0)

# --- Op Definitions (Integrated) ---

struct Op(CollectionElement):
    var tag: Int
    var lhs: List[Tensor]
    var bin_op: Int
    var un_op: Int
    var dim: List[Int]

    fn __init__(inout self):
        self.tag = 0
        self.lhs = List[Tensor]()
        self.bin_op = 0
        self.un_op = 0
        self.dim = List[Int]()

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

    @staticmethod
    fn reshape(arg: Tensor) -> Op:
        var o = Op()
        o.tag = 4
        o.lhs.append(arg)
        return o

    @staticmethod
    fn transpose(arg: Tensor) -> Op:
        var o = Op()
        o.tag = 5
        o.lhs.append(arg)
        return o

    @staticmethod
    fn broadcast(arg: Tensor) -> Op:
        var o = Op()
        o.tag = 6
        o.lhs.append(arg)
        return o

    @staticmethod
    fn reduce(arg: Tensor, op: Int, dims: List[Int]) -> Op:
        var o = Op()
        o.tag = 7
        o.lhs.append(arg)
        o.un_op = op
        o.dim = dims
        return o

    fn __copyinit__(inout self, other: Op):
        self.tag = other.tag
        self.lhs = other.lhs
        self.bin_op = other.bin_op
        self.un_op = other.un_op
        self.dim = other.dim

    fn __moveinit__(inout self, other: Op):
        self.tag = other.tag
        self.lhs = other.lhs
        self.bin_op = other.bin_op
        self.un_op = other.un_op
        self.dim = other.dim

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

# --- Tensor Struct ---

struct Tensor(CollectionElement, Stringable, EqualityComparable):
    var _storage: Rc[Storage]
    var _layout: Layout
    var _id: Int
    var _op: BackpropOp
    var _is_variable: Bool

    fn __init__(inout self, storage: Rc[Storage], layout: Layout, op: BackpropOp, is_variable: Bool):
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
        return self._storage[].dtype

    fn device(self) -> Device:
        return self._storage[].device

    fn layout(self) -> Layout:
        return self._layout

    fn id(self) -> Int:
        return self._id

    fn is_variable(self) -> Bool:
        return self._is_variable

    fn __str__(self) -> String:
        return "Tensor[" + str(self.shape()) + ", " + self.dtype().as_str() + "]"

    # --- Creation ---

    @staticmethod
    fn from_storage(storage: Rc[Storage], layout: Layout, op: BackpropOp, is_variable: Bool) -> Tensor:
        return Tensor(storage, layout, op, is_variable)

    @staticmethod
    fn new(data: List[Float32], shape: Shape, device: Device) raises -> Tensor:
        if shape.elem_count() != len(data):
            raise Error("Shape size does not match data length")
        var storage = Rc[Storage](Storage(data, DType.F32, device))
        var layout = Layout.contiguous(shape)
        return Tensor(storage, layout, BackpropOp.none(), False)

    @staticmethod
    fn zeros(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var storage = Rc[Storage](Storage(count, dtype, device))
        # Zero init handled in Storage init
        return Tensor(storage, Layout.contiguous(shape), BackpropOp.none(), False)

    @staticmethod
    fn ones(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var storage = Rc[Storage](Storage(count, dtype, device))
        var ptr = storage.get_mut()[].data
        for i in range(count):
            ptr[i] = 1.0
        return Tensor(storage, Layout.contiguous(shape), BackpropOp.none(), False)

    fn zeros_like(self) raises -> Tensor:
        return Tensor.zeros(self.shape(), self.dtype(), self.device())

    fn ones_like(self) raises -> Tensor:
        return Tensor.ones(self.shape(), self.dtype(), self.device())

    # --- View Ops ---

    fn reshape(self, shape: Shape) raises -> Tensor:
        if shape.elem_count() != self.shape().elem_count():
             raise Error("Shape mismatch in reshape")
        if not self._layout.is_contiguous():
             raise Error("Reshape currently only supports contiguous tensors")
        var new_layout = Layout.contiguous(shape)
        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            op = BackpropOp(Op.reshape(self))
        return Tensor(self._storage, new_layout, op, False)

    fn transpose(self, dim1: Int, dim2: Int) raises -> Tensor:
        var new_layout = self._layout.transpose(dim1, dim2)
        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            op = BackpropOp(Op.transpose(self))
        return Tensor(self._storage, new_layout, op, False)

    fn narrow(self, dim: Int, start: Int, length: Int) raises -> Tensor:
        var new_layout = self._layout.narrow(dim, start, length)
        return Tensor(self._storage, new_layout, BackpropOp.none(), False)

    fn broadcast_as(self, shape: Shape) raises -> Tensor:
        var new_layout = self._layout.broadcast_as(shape)
        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            op = BackpropOp(Op.broadcast(self))
        return Tensor(self._storage, new_layout, op, False)

    # --- Ops ---

    fn flatten_to_list(self) -> List[Float32]:
        # Using pointer access now
        # Rc[].data returns pointer
        var ptr = self._storage[].data

        if self._layout.is_contiguous():
            var start = self._layout.start_offset()
            var count = self._layout.shape().elem_count()
            var res = List[Float32](capacity=count)
            for i in range(count):
                res.append(ptr[start + i])
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
            res.append(ptr[current_offset])
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

        var out_storage = Rc[Storage](Storage(m*n, self.dtype(), self.device()))
        var out_ptr = out_storage.get_mut()[].data

        for i in range(m):
            for j in range(n):
                var sum: Float32 = 0.0
                for l in range(k):
                    var val_a = lhs_data[i * k + l]
                    var val_b = rhs_data[l * n + j]
                    sum += val_a * val_b
                out_ptr[i * n + j] = sum

        var op = BackpropOp.none()
        if self._is_variable or rhs._is_variable or self._op.has_op or rhs._op.has_op:
            op = BackpropOp(Op.matmul(self, rhs))

        return Tensor(out_storage, Layout.contiguous(res_shape), op, False)

    fn _binary_op(self, rhs: Tensor, op_code: Int) raises -> Tensor:
        var shape = self.shape().broadcast_shape_binary_op(rhs.shape())
        var lhs_b = self.broadcast_as(shape)
        var rhs_b = rhs.broadcast_as(shape)

        var lhs_data = lhs_b.flatten_to_list()
        var rhs_data = rhs_b.flatten_to_list()
        var count = shape.elem_count()

        var out_storage = Rc[Storage](Storage(count, self.dtype(), self.device()))
        var out_ptr = out_storage.get_mut()[].data

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
            elif op_code == BinOpEnum.Max:
                val = a if a > b else b
            out_ptr[i] = val

        var op = BackpropOp.none()
        if self._is_variable or rhs._is_variable or self._op.has_op or rhs._op.has_op:
            op = BackpropOp(Op.binary(self, rhs, op_code))

        return Tensor(out_storage, Layout.contiguous(shape), op, False)

    fn add(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Add)

    fn sub(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Sub)

    fn mul(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Mul)

    fn div(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Div)

    fn maximum(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, BinOpEnum.Max)

    fn _unary_op(self, op_code: Int) raises -> Tensor:
        var data = self.flatten_to_list()
        var count = len(data)

        var out_storage = Rc[Storage](Storage(count, self.dtype(), self.device()))
        var out_ptr = out_storage.get_mut()[].data

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
            elif op_code == UnOpEnum.Relu:
                r = v if v > 0.0 else 0.0
            elif op_code == UnOpEnum.Gelu:
                var c1 = 0.7978845608
                var c2 = 0.044715
                var x3 = v * v * v
                var inner = c1 * (v + c2 * x3)
                r = 0.5 * v * (1.0 + math.tanh(inner))
            out_ptr[i] = r

        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            op = BackpropOp(Op.unary(self, op_code))

        return Tensor(out_storage, Layout.contiguous(self.shape()), op, False)

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

    fn relu(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Relu)

    fn gelu(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Gelu)

    # --- Reductions ---

    fn sum(self, dim: Int) raises -> Tensor:
        var shape = self.shape()
        var dims = shape.dims
        if dim >= len(dims): raise Error("Dim out of range")

        var stride = self._layout.stride()
        var dim_stride = stride[dim]
        var dim_size = dims[dim]

        var new_dims = List[Int]()
        for i in range(len(dims)):
            if i != dim: new_dims.append(dims[i])
        var new_shape = Shape(new_dims)
        var new_count = new_shape.elem_count()

        var out_storage = Rc[Storage](Storage(new_count, self.dtype(), self.device()))
        var out_ptr = out_storage.get_mut()[].data

        var flat_data = self.flatten_to_list()
        var old_stride = shape.stride_contiguous()

        for i in range(new_count):
            var coords = List[Int](capacity=len(new_dims))
            var temp = i
            for d in range(len(new_dims)):
                coords.append(0)

            for d in range(len(new_dims)-1, -1, -1):
                coords[d] = temp % new_dims[d]
                temp = temp // new_dims[d]

            var base_idx = 0
            var c_idx = 0
            for d in range(len(dims)):
                if d == dim:
                    continue
                base_idx += coords[c_idx] * old_stride[d]
                c_idx += 1

            var sum_val: Float32 = 0.0
            for k in range(dim_size):
                sum_val += flat_data[base_idx + k * old_stride[dim]]
            out_ptr[i] = sum_val

        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            var dims_arg = List[Int]()
            dims_arg.append(dim)
            op = BackpropOp(Op.reduce(self, 0, dims_arg)) # 0 for sum

        return Tensor(out_storage, Layout.contiguous(new_shape), op, False)

    fn mean(self, dim: Int) raises -> Tensor:
        var sum_t = self.sum(dim)
        var dim_size = self.shape().dims[dim]
        var scale_data = List[Float32]()
        scale_data.append(1.0 / Float32(dim_size))
        var scale = Tensor.new(scale_data, Shape(1), self.device())
        return sum_t.mul(scale)

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
                        self._accumulate_grad(grads, arg, grad.mul(node))
                    elif code == UnOpEnum.Sqr:
                        var two_data = List[Float32]()
                        two_data.append(2.0)
                        var two = Tensor.new(two_data, Shape(1), arg.device())
                        var two_arg = arg.mul(two)
                        self._accumulate_grad(grads, arg, grad.mul(two_arg))
                    elif code == UnOpEnum.Relu:
                        var mask_count = arg.shape().elem_count()
                        var mask_storage = Rc[Storage](Storage(mask_count, arg.dtype(), arg.device()))
                        var mask_ptr = mask_storage.get_mut()[].data
                        var arg_data = arg.flatten_to_list()
                        for i in range(len(arg_data)):
                            mask_ptr[i] = 1.0 if arg_data[i] > 0.0 else 0.0
                        var mask = Tensor(mask_storage, Layout.contiguous(arg.shape()), BackpropOp.none(), False)
                        self._accumulate_grad(grads, arg, grad.mul(mask))

                elif op.tag == 4: # Reshape
                    var arg = op.lhs[0]
                    var arg_grad = grad.reshape(arg.shape())
                    self._accumulate_grad(grads, arg, arg_grad)

                elif op.tag == 5: # Transpose
                    var arg = op.lhs[0]
                    var arg_grad = grad.transpose(0, 1)
                    self._accumulate_grad(grads, arg, arg_grad)

                elif op.tag == 6: # Broadcast
                    var arg = op.lhs[0]
                    if arg.shape().rank() < grad.shape().rank():
                        var diff = grad.shape().rank() - arg.shape().rank()
                        var reduced = grad
                        for _ in range(diff):
                            reduced = reduced.sum(0)
                        self._accumulate_grad(grads, arg, reduced)
                    else:
                        var curr_grad = grad
                        for d in range(arg.shape().rank()):
                            if arg.shape().dims[d] == 1 and grad.shape().dims[d] > 1:
                                curr_grad = curr_grad.sum(d)
                        self._accumulate_grad(grads, arg, curr_grad)

                elif op.tag == 7: # Reduce (Sum)
                    var arg = op.lhs[0]
                    var dim = op.dim[0]
                    # Gradient of sum is broadcast of gradient
                    # grad is [d0, ..., d_reduced, ...]
                    # arg is [d0, ..., d_dim, ...]
                    # We need to unsqueeze grad at dim, then broadcast to arg.
                    # 1. Unsqueeze (Reshape insert 1 at dim)
                    var g_shape = grad.shape().dims
                    var new_dims = List[Int]()
                    for i in range(dim):
                        new_dims.append(g_shape[i])
                    new_dims.append(1)
                    for i in range(dim, len(g_shape)):
                        new_dims.append(g_shape[i])

                    var g_unsqueezed = grad.reshape(Shape(new_dims))
                    var g_broadcast = g_unsqueezed.broadcast_as(arg.shape())
                    self._accumulate_grad(grads, arg, g_broadcast)

        return grads

    fn _accumulate_grad(self, inout grads: Grads, target: Tensor, g: Tensor) raises:
        for i in range(len(grads.ids)):
            if grads.ids[i] == target._id:
                grads.grads[i] = grads.grads[i].add(g)
                return
        grads.ids.append(target._id)
        grads.grads.append(g)
