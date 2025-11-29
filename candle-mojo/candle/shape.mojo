from collections import List

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
            count *= d[]
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
