from collections import List
from .dtype import DType
from .device import Device
from .shape import Shape
from .error import Error
from .layout import Layout
import math

# Simplistic Storage implementation to hold data
struct Storage(CollectionElement):
    var data: List[Float32] # Simplified to F32 for now
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

struct Tensor(CollectionElement, Stringable):
    var _storage: Storage
    var _layout: Layout

    fn __init__(inout self, storage: Storage, layout: Layout):
        self._storage = storage
        self._layout = layout

    fn __copyinit__(inout self, other: Tensor):
        self._storage = other._storage
        self._layout = other._layout

    fn __moveinit__(inout self, other: Tensor):
        self._storage = other._storage
        self._layout = other._layout

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

    fn __str__(self) -> String:
        return "Tensor[" + str(self.shape()) + ", " + self.dtype().as_str() + "]"

    # --- Creation methods ---

    @staticmethod
    fn new(data: List[Float32], shape: Shape, device: Device) raises -> Tensor:
        if shape.elem_count() != len(data):
            raise Error("Shape size does not match data length")
        var storage = Storage(data, DType.F32, device)
        var layout = Layout.contiguous(shape)
        return Tensor(storage, layout)

    @staticmethod
    fn zeros(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for _ in range(count):
            data.append(0.0)
        return Tensor(Storage(data, dtype, device), Layout.contiguous(shape))

    @staticmethod
    fn ones(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for _ in range(count):
            data.append(1.0)
        return Tensor(Storage(data, dtype, device), Layout.contiguous(shape))

    @staticmethod
    fn randn(mean: Float32, std: Float32, shape: Shape, device: Device) -> Tensor:
        # Placeholder for random generation
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for i in range(count):
            data.append(mean) # TODO: Actual random
        return Tensor(Storage(data, DType.F32, device), Layout.contiguous(shape))

    # --- Indexing and Movement ---

    fn reshape(self, shape: Shape) raises -> Tensor:
        if shape.elem_count() != self.shape().elem_count():
             raise Error("Shape mismatch in reshape")
        if not self._layout.is_contiguous():
             raise Error("Reshape currently only supports contiguous tensors")
        return Tensor(self._storage, Layout.contiguous(shape))

    fn transpose(self, dim1: Int, dim2: Int) raises -> Tensor:
        var new_layout = self._layout.transpose(dim1, dim2)
        return Tensor(self._storage, new_layout)

    fn narrow(self, dim: Int, start: Int, length: Int) raises -> Tensor:
        var new_layout = self._layout.narrow(dim, start, length)
        return Tensor(self._storage, new_layout)

    fn broadcast_as(self, shape: Shape) raises -> Tensor:
        var new_layout = self._layout.broadcast_as(shape)
        return Tensor(self._storage, new_layout)

    # --- Operations ---

    # Helper to iterate over tensor elements based on layout
    # Since we don't have a complex iterator yet, we'll flatten to a list for ops
    # This is inefficient (copies data) but correct for behavior
    fn flatten_to_list(self) -> List[Float32]:
        # If contiguous, we can just slice the storage (taking offset into account)
        if self._layout.is_contiguous():
            var start = self._layout.start_offset()
            var count = self._layout.shape().elem_count()
            var res = List[Float32](capacity=count)
            for i in range(count):
                res.append(self._storage.data[start + i])
            return res

        # General case: recursive iteration or coordinate calculation
        # Implementing a simple coordinate counter
        var shape = self.shape()
        var dims = shape.dims
        var stride = self._layout.stride()
        var offset = self._layout.start_offset()
        var count = shape.elem_count()
        var res = List[Float32](capacity=count)

        # Iteration logic for arbitrary rank is tricky without recursion or a multi-index
        # Using a flat index and converting to coords
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
        # Basic matrix multiplication
        # We'll enforce contiguous input for now to simplify inner loop,
        # or just use the flatten_to_list which handles it (slowly).

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

        # Materialize data
        var lhs_data = self.flatten_to_list()
        var rhs_data = rhs.flatten_to_list()

        # Result shape (simplest 2D case - disregarding batch dims for this simple port)
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

        return Tensor(Storage(res_data, self.dtype(), self.device()), Layout.contiguous(res_shape))

    # Generic Binary Op
    fn _binary_op(self, rhs: Tensor, op_code: Int) raises -> Tensor:
        # op_code: 0=add, 1=sub, 2=mul, 3=div

        # Broadcast logic
        var shape = self.shape().broadcast_shape_binary_op(rhs.shape())

        # Broadcast inputs
        # Note: broadcast_as is effectively a view, so cheap
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
            if op_code == 0:
                val = a + b
            elif op_code == 1:
                val = a - b
            elif op_code == 2:
                val = a * b
            elif op_code == 3:
                val = a / b
            res_data.append(val)

        return Tensor(Storage(res_data, self.dtype(), self.device()), Layout.contiguous(shape))

    fn add(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, 0)

    fn sub(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, 1)

    fn mul(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, 2)

    fn div(self, rhs: Tensor) raises -> Tensor:
        return self._binary_op(rhs, 3)

    # Generic Unary Op
    fn _unary_op(self, op_code: Int) raises -> Tensor:
        # op_code: 0=neg, 1=exp, 2=log, 3=sqr, 4=sqrt, 5=recip, 6=sin, 7=cos
        var data = self.flatten_to_list()
        var count = len(data)
        var res_data = List[Float32](capacity=count)

        for i in range(count):
            var v = data[i]
            var r: Float32 = 0.0
            if op_code == 0:
                r = -v
            elif op_code == 1:
                r = math.exp(v)
            elif op_code == 2:
                r = math.log(v)
            elif op_code == 3:
                r = v * v
            elif op_code == 4:
                r = math.sqrt(v)
            elif op_code == 5:
                r = 1.0 / v
            elif op_code == 6:
                r = math.sin(v)
            elif op_code == 7:
                r = math.cos(v)
            res_data.append(r)

        return Tensor(Storage(res_data, self.dtype(), self.device()), Layout.contiguous(self.shape()))

    fn neg(self) raises -> Tensor:
        return self._unary_op(0)

    fn exp(self) raises -> Tensor:
        return self._unary_op(1)

    fn log(self) raises -> Tensor:
        return self._unary_op(2)

    fn sqr(self) raises -> Tensor:
        return self._unary_op(3)

    fn sqrt(self) raises -> Tensor:
        return self._unary_op(4)

    fn recip(self) raises -> Tensor:
        return self._unary_op(5)

    fn sin(self) raises -> Tensor:
        return self._unary_op(6)

    fn cos(self) raises -> Tensor:
        return self._unary_op(7)
