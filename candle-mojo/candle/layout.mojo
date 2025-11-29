from collections import List
from .shape import Shape
from .error import Error

struct Layout(CollectionElement, Stringable):
    var _shape: Shape
    var _stride: List[Int]
    var _start_offset: Int

    fn __init__(inout self, shape: Shape, stride: List[Int], start_offset: Int):
        self._shape = shape
        self._stride = stride
        self._start_offset = start_offset

    fn __copyinit__(inout self, other: Layout):
        self._shape = other._shape
        self._stride = other._stride
        self._start_offset = other._start_offset

    fn __moveinit__(inout self, other: Layout):
        self._shape = other._shape
        self._stride = other._stride
        self._start_offset = other._start_offset

    @staticmethod
    fn contiguous(shape: Shape) -> Layout:
        return Layout(shape, shape.stride_contiguous(), 0)

    fn shape(self) -> Shape:
        return self._shape

    fn stride(self) -> List[Int]:
        return self._stride

    fn start_offset(self) -> Int:
        return self._start_offset

    fn dims(self) -> List[Int]:
        return self._shape.dims

    fn is_contiguous(self) -> Bool:
        return self._shape.is_contiguous(self._stride)

    fn transpose(self, dim1: Int, dim2: Int) raises -> Layout:
        var rank = self._shape.rank()
        if dim1 >= rank or dim2 >= rank:
             raise Error("Dimension out of range in transpose")

        var new_dims = self._shape.dims
        # Swap dims
        var tmp_d = new_dims[dim1]
        new_dims[dim1] = new_dims[dim2]
        new_dims[dim2] = tmp_d

        var new_stride = self._stride
        # Swap strides
        var tmp_s = new_stride[dim1]
        new_stride[dim1] = new_stride[dim2]
        new_stride[dim2] = tmp_s

        return Layout(Shape(new_dims), new_stride, self._start_offset)

    fn narrow(self, dim: Int, start: Int, length: Int) raises -> Layout:
        var dims = self._shape.dims
        if dim >= len(dims):
            raise Error("Dim out of range in narrow")
        if start + length > dims[dim]:
            raise Error("Narrow range out of bounds")

        var new_dims = dims
        new_dims[dim] = length

        var new_offset = self._start_offset + self._stride[dim] * start

        return Layout(Shape(new_dims), self._stride, new_offset)

    fn broadcast_as(self, shape: Shape) raises -> Layout:
        if shape.rank() < self._shape.rank():
            raise Error("Broadcast: target rank is smaller than source rank")

        var added_dims = shape.rank() - self._shape.rank()
        var new_stride = List[Int]()

        # Fill added dims with 0 stride
        for _ in range(added_dims):
            new_stride.append(0)

        # Check compatibility for existing dims
        for i in range(self._shape.rank()):
            var src_dim = self._shape.dims[i]
            var dst_dim = shape.dims[added_dims + i]
            var src_stride = self._stride[i]

            if src_dim == dst_dim:
                new_stride.append(src_stride)
            elif src_dim == 1:
                new_stride.append(0)
            else:
                raise Error("Broadcast: incompatible dimensions")

        return Layout(shape, new_stride, self._start_offset)

    fn __str__(self) -> String:
        return "Layout(shape=" + str(self._shape) + ", stride=" + str(self._stride) + ", offset=" + str(self._start_offset) + ")"
