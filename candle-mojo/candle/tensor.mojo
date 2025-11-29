from collections import List
from .dtype import DType
from .device import Device
from .shape import Shape
from .error import Error

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
    var _shape: Shape
    var _stride: List[Int]

    fn __init__(inout self, storage: Storage, shape: Shape):
        self._storage = storage
        self._shape = shape
        self._stride = shape.stride_contiguous()

    fn __copyinit__(inout self, other: Tensor):
        self._storage = other._storage
        self._shape = other._shape
        self._stride = other._stride

    fn __moveinit__(inout self, other: Tensor):
        self._storage = other._storage
        self._shape = other._shape
        self._stride = other._stride

    fn shape(self) -> Shape:
        return self._shape

    fn dtype(self) -> DType:
        return self._storage.dtype

    fn device(self) -> Device:
        return self._storage.device

    fn __str__(self) -> String:
        return "Tensor[" + str(self._shape) + ", " + self.dtype().as_str() + "]"

    # --- Creation methods ---

    @staticmethod
    fn new(data: List[Float32], shape: Shape, device: Device) raises -> Tensor:
        if shape.elem_count() != len(data):
            raise Error("Shape size does not match data length")
        var storage = Storage(data, DType.F32, device)
        return Tensor(storage, shape)

    @staticmethod
    fn zeros(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for _ in range(count):
            data.append(0.0)
        return Tensor(Storage(data, dtype, device), shape)

    @staticmethod
    fn ones(shape: Shape, dtype: DType, device: Device) -> Tensor:
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for _ in range(count):
            data.append(1.0)
        return Tensor(Storage(data, dtype, device), shape)

    @staticmethod
    fn randn(mean: Float32, std: Float32, shape: Shape, device: Device) -> Tensor:
        # Placeholder for random generation
        var count = shape.elem_count()
        var data = List[Float32](capacity=count)
        for i in range(count):
            # Using a pseudo-random generator or just filling for now
            # In a real port, we'd bind to system random or use a seed
            data.append(mean)
        return Tensor(Storage(data, DType.F32, device), shape)

    # --- Operations ---

    fn reshape(self, shape: Shape) raises -> Tensor:
        if shape.elem_count() != self._shape.elem_count():
             raise Error("Shape mismatch in reshape")
        return Tensor(self._storage, shape)

    fn matmul(self, rhs: Tensor) raises -> Tensor:
        # Basic matrix multiplication check
        var lhs_dims = self._shape.dims
        var rhs_dims = rhs._shape.dims

        if len(lhs_dims) < 2 or len(rhs_dims) < 2:
             raise Error("Matmul requires at least 2 dimensions")

        var m = lhs_dims[len(lhs_dims) - 2]
        var k = lhs_dims[len(lhs_dims) - 1]
        var k2 = rhs_dims[len(rhs_dims) - 2]
        var n = rhs_dims[len(rhs_dims) - 1]

        if k != k2:
             raise Error("Matmul dimension mismatch")

        # Result shape (simplest 2D case)
        var res_shape = Shape(m, n)
        var res_data = List[Float32](capacity=m*n)
        for _ in range(m*n):
            res_data.append(0.0)

        # Very naive implementation for proof of concept
        # Accessing data directly via lists (slow, but functionally demonstrating logic)
        for i in range(m):
            for j in range(n):
                var sum: Float32 = 0.0
                for l in range(k):
                    var val_a = self._storage.data[i * k + l]
                    var val_b = rhs._storage.data[l * n + j]
                    sum += val_a * val_b
                res_data[i * n + j] = sum

        return Tensor(Storage(res_data, self.dtype(), self.device()), res_shape)

    fn add(self, rhs: Tensor) raises -> Tensor:
         # Basic element-wise add
         if self._shape.elem_count() != rhs._shape.elem_count():
             # For now, require strict shape match (no broadcasting)
             raise Error("Shape mismatch in add")

         var count = self._shape.elem_count()
         var res_data = List[Float32](capacity=count)

         for i in range(count):
             res_data.append(self._storage.data[i] + rhs._storage.data[i])

         return Tensor(Storage(res_data, self.dtype(), self.device()), self._shape)
