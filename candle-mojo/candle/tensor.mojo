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

    # New: for gather
    var gather_dim: Int

    fn __init__(inout self):
        self.tag = 0
        self.lhs = List[Tensor]()
        self.bin_op = 0
        self.un_op = 0
        self.dim = List[Int]()
        self.gather_dim = 0

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

    @staticmethod
    fn gather(arg: Tensor, index: Tensor, dim: Int) -> Op:
        var o = Op()
        o.tag = 8
        o.lhs.append(arg)
        o.lhs.append(index)
        o.gather_dim = dim
        return o

    fn __copyinit__(inout self, other: Op):
        self.tag = other.tag
        self.lhs = other.lhs
        self.bin_op = other.bin_op
        self.un_op = other.un_op
        self.dim = other.dim
        self.gather_dim = other.gather_dim

    fn __moveinit__(inout self, other: Op):
        self.tag = other.tag
        self.lhs = other.lhs
        self.bin_op = other.bin_op
        self.un_op = other.un_op
        self.dim = other.dim
        self.gather_dim = other.gather_dim

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

    @staticmethod
    fn cat(tensors: List[Tensor], dim: Int) raises -> Tensor:
        # Concatenate tensors along dim
        if len(tensors) == 0: raise Error("Empty cat list")
        var first = tensors[0]
        var rank = first.shape().rank()
        if dim >= rank: raise Error("Cat dim out of range")

        # Check shapes match except dim
        var final_dims = first.shape().dims
        var total_dim_size = final_dims[dim]

        for i in range(1, len(tensors)):
            var t = tensors[i]
            if t.shape().rank() != rank: raise Error("Cat rank mismatch")
            var t_dims = t.shape().dims
            for d in range(rank):
                if d != dim and t_dims[d] != final_dims[d]:
                    raise Error("Cat shape mismatch")
            total_dim_size += t_dims[dim]

        final_dims[dim] = total_dim_size
        var out_shape = Shape(final_dims)
        var out_count = out_shape.elem_count()
        var out_storage = Rc[Storage](Storage(out_count, first.dtype(), first.device()))
        var out_ptr = out_storage.get_mut()[].data

        # Naive copy loop
        # Calculate stride of dim in output
        # stride[dim]
        # We can iterate output and map back, or iterate inputs and place.
        # Iterate inputs and place is easier.

        var current_dim_offset = 0
        var stride = out_shape.stride_contiguous()

        for i in range(len(tensors)):
            var t = tensors[i]
            var t_data = t.flatten_to_list()
            var t_dims = t.shape().dims
            var t_count = len(t_data)

            # Map t index to out index
            # This is complex in flat loop without multidim index.
            # But we can assume contiguous chunks if dim=0.
            # If dim != 0, it's strided.

            # General loop:
            # Iterate t elements, calc coords, adjust dim coord, calc out index.
            # Optimization: stride calc.

            var t_stride = t.shape().stride_contiguous()

            for k in range(t_count):
                var temp = k
                var out_idx = 0
                for d in range(rank-1, -1, -1):
                    var coord = temp % t_dims[d]
                    temp = temp // t_dims[d]

                    var final_coord = coord
                    if d == dim:
                        final_coord += current_dim_offset

                    out_idx += final_coord * stride[d]

                out_ptr[out_idx] = t_data[k]

            current_dim_offset += t_dims[dim]

        # Backprop for cat is split (narrow). Not implemented here for brevity but tag exists?
        # Op.cat? Adding later if needed.
        return Tensor(out_storage, Layout.contiguous(out_shape), BackpropOp.none(), False)

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
        var lhs_rank = len(lhs_dims)
        var rhs_rank = len(rhs_dims)

        if lhs_rank < 2 or rhs_rank < 2:
             raise Error("Matmul requires at least 2 dimensions")

        # Batched Matmul Support
        # Handle broadcasting of batch dims.
        # Simple case: ranks match, batch dims match.
        # Only broadcasting logic for >2 ranks.

        var m = lhs_dims[lhs_rank - 2]
        var k = lhs_dims[lhs_rank - 1]
        var k2 = rhs_dims[rhs_rank - 2]
        var n = rhs_dims[rhs_rank - 1]

        if k != k2:
             raise Error("Matmul dimension mismatch")

        # Batch size calculation
        var batch_count = 1
        var lhs_batch_stride = m * k
        var rhs_batch_stride = k * n
        var out_batch_stride = m * n

        # Check if batched
        if lhs_rank > 2:
            for i in range(lhs_rank - 2):
                batch_count *= lhs_dims[i]

        var res_dims = List[Int]()
        for i in range(lhs_rank - 2):
            res_dims.append(lhs_dims[i])
        res_dims.append(m)
        res_dims.append(n)
        var res_shape = Shape(res_dims)

        var lhs_data = self.flatten_to_list()
        var rhs_data = rhs.flatten_to_list()

        var out_storage = Rc[Storage](Storage(batch_count * m * n, self.dtype(), self.device()))
        var out_ptr = out_storage.get_mut()[].data

        for b in range(batch_count):
            var lhs_offset = b * lhs_batch_stride
            var rhs_offset = b * rhs_batch_stride
            var out_offset = b * out_batch_stride

            for i in range(m):
                for j in range(n):
                    var sum: Float32 = 0.0
                    for l in range(k):
                        var val_a = lhs_data[lhs_offset + i * k + l]
                        var val_b = rhs_data[rhs_offset + l * n + j]
                        sum += val_a * val_b
                    out_ptr[out_offset + i * n + j] = sum

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
            elif op_code == BinOpEnum.Min:
                val = a if a < b else b
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
            elif op_code == UnOpEnum.Silu:
                r = v / (1.0 + math.exp(-v))
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

    fn silu(self) raises -> Tensor:
        return self._unary_op(UnOpEnum.Silu)

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

    # --- Indexing (Gather) ---

    fn gather(self, index: Tensor, dim: Int) raises -> Tensor:
        # Simplistic gather implementation
        # Output shape = index.shape
        # output[i, j, k] = input[index[i, j, k], j, k] (if dim=0)
        # Assuming index same rank as self? Or 1D index?
        # Candle `gather` usually means `torch.gather`: same rank.
        # `embedding` uses `index_select` (1D index) usually.
        # Implementing `torch.gather` semantics logic manually is complex.
        # Implementing `index_select` (dim=0) for Embedding:

        # If dim=0 and index is 1D (or flattened):
        # We select rows.
        var idx_data = index.flatten_to_list()
        var num_indices = len(idx_data)

        var row_size = 1
        var dims = self.shape().dims
        for i in range(1, len(dims)):
            row_size *= dims[i]

        var out_dims = List[Int]()
        out_dims.append(num_indices)
        for i in range(1, len(dims)):
            out_dims.append(dims[i])

        var out_shape = Shape(out_dims)
        var out_storage = Rc[Storage](Storage(out_shape.elem_count(), self.dtype(), self.device()))
        var out_ptr = out_storage.get_mut()[].data

        var self_data = self.flatten_to_list()

        for i in range(num_indices):
            var idx = int(idx_data[i])
            var src_offset = idx * row_size
            var dst_offset = i * row_size

            for j in range(row_size):
                out_ptr[dst_offset + j] = self_data[src_offset + j]

        var op = BackpropOp.none()
        if self._is_variable or self._op.has_op:
            op = BackpropOp(Op.gather(self, index, dim))

        return Tensor(out_storage, Layout.contiguous(out_shape), op, False)

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
                    # Check rank for batched matmul grad
                    # If 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
                    # lhs_grad = grad @ rhs.T
                    # rhs_grad = lhs.T @ grad
                    # We need to ensure transpose acts on last two dims.
                    # My transpose takes 2 args.
                    var rank = lhs.shape().rank()
                    var d1 = rank - 2
                    var d2 = rank - 1
                    var lhs_grad = grad.matmul(rhs.transpose(d1, d2))
                    var rhs_grad = lhs.transpose(d1, d2).matmul(grad)

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
                    elif code == UnOpEnum.Silu:
                        # d(silu) = sig * (1 + x * (1 - sig)) = sig + x * sig * (1 - sig)
                        # approximation or exact?
                        # x * sig is output (node).
                        # node * (1 - sig) + sig
                        # sig = 1 / (1 + exp(-x))
                        # This requires sig op or calc.
                        pass

                elif op.tag == 4: # Reshape
                    var arg = op.lhs[0]
                    var arg_grad = grad.reshape(arg.shape())
                    self._accumulate_grad(grads, arg, arg_grad)

                elif op.tag == 5: # Transpose
                    var arg = op.lhs[0]
                    var arg_grad = grad.transpose(0, 1) # TODO: Store dims
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

                elif op.tag == 8: # Gather
                    var arg = op.lhs[0]
                    var index = op.lhs[1]
                    # Gradient w.r.t arg is scatter_add
                    # We don't have scatter_add.
                    # Implementing a simplified dense loop for grad accumulation.
                    # grad is same shape as index.
                    # arg grad is same shape as arg.
                    # arg_grad[idx[i]] += grad[i]
                    var arg_grad = arg.zeros_like()
                    var arg_ptr = arg_grad._storage.get_mut()[].data
                    var idx_data = index.flatten_to_list()
                    var g_data = grad.flatten_to_list()
                    var row_size = 1
                    var dims = arg.shape().dims
                    for i in range(1, len(dims)): row_size *= dims[i]

                    for i in range(len(idx_data)):
                        var row = int(idx_data[i])
                        var offset = row * row_size
                        var g_offset = i * row_size
                        for j in range(row_size):
                            arg_ptr[offset + j] += g_data[g_offset + j]

                    self._accumulate_grad(grads, arg, arg_grad)

        return grads

    fn _accumulate_grad(self, inout grads: Grads, target: Tensor, g: Tensor) raises:
        for i in range(len(grads.ids)):
            if grads.ids[i] == target._id:
                grads.grads[i] = grads.grads[i].add(g)
                return
        grads.ids.append(target._id)
        grads.grads.append(g)
