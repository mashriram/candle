from collections import List
from .error import Error

struct Shape(CollectionElement, Sized, Stringable):
    var dims: List[Int]

    fn __init__(inout self, dims: List[Int]):
        self.dims = dims

    fn __init__(inout self, *dims: Int):
        self.dims = List[Int]()
        for d in dims:
            self.dims.append(d)

    fn __copyinit__(inout self, other: Shape):
        self.dims = other.dims

    fn __moveinit__(inout self, other: Shape):
        self.dims = other.dims

    fn __str__(self) -> String:
        var s = String("[")
        for i in range(len(self.dims)):
            s += str(self.dims[i])
            if i < len(self.dims) - 1:
                s += ", "
        s += "]"
        return s

    fn __len__(self) -> Int:
        return len(self.dims)

    fn __getitem__(self, idx: Int) -> Int:
        return self.dims[idx]

    fn rank(self) -> Int:
        return len(self.dims)

    fn elem_count(self) -> Int:
        if len(self.dims) == 0:
            return 1 # Scalar
        var count = 1
        for d in self.dims:
            count *= d
        return count

    fn stride_contiguous(self) -> List[Int]:
        var stride = List[Int]()
        if len(self.dims) == 0:
            return stride

        # Initialize with size of dims
        for _ in range(len(self.dims)):
            stride.append(1)

        var acc = 1
        for i in range(len(self.dims) - 1, -1, -1):
            stride[i] = acc
            acc *= self.dims[i]

        return stride

    fn is_contiguous(self, stride: List[Int]) -> Bool:
        if len(self.dims) != len(stride):
            return False
        var acc = 1
        for i in range(len(self.dims) - 1, -1, -1):
            var dim = self.dims[i]
            var s = stride[i]
            if dim > 1 and s != acc:
                return False
            acc *= dim
        return True

    fn broadcast_shape_binary_op(self, rhs: Shape) raises -> Shape:
        var lhs_dims = self.dims
        var rhs_dims = rhs.dims
        var lhs_ndims = len(lhs_dims)
        var rhs_ndims = len(rhs_dims)
        var bcast_ndims = max(lhs_ndims, rhs_ndims)

        var bcast_dims = List[Int](capacity=bcast_ndims)
        for _ in range(bcast_ndims):
            bcast_dims.append(0)

        for idx in range(bcast_ndims):
            var rev_idx = bcast_ndims - idx
            var l_value = 1
            if lhs_ndims >= rev_idx:
                l_value = lhs_dims[lhs_ndims - rev_idx]
            var r_value = 1
            if rhs_ndims >= rev_idx:
                r_value = rhs_dims[rhs_ndims - rev_idx]

            if l_value != r_value and l_value != 1 and r_value != 1:
                raise Error("Shape mismatch in broadcast")

            if l_value == r_value:
                bcast_dims[idx] = l_value
            elif l_value == 1:
                bcast_dims[idx] = r_value
            else:
                bcast_dims[idx] = l_value

        return Shape(bcast_dims)
