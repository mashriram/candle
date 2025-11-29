from .dtype import DType
from .shape import Shape

struct Device(CollectionElement, Stringable, EqualityComparable):
    var _type: Int # 0 for CPU, 1 for CUDA, 2 for Metal
    var _id: Int

    alias CPU = Device(0, 0)

    # Static factory methods to mimic Rust enums
    @staticmethod
    fn new_cuda(ordinal: Int) -> Device:
        return Device(1, ordinal)

    @staticmethod
    fn new_metal(ordinal: Int) -> Device:
        return Device(2, ordinal)

    fn __init__(inout self, type: Int, id: Int):
        self._type = type
        self._id = id

    fn __copyinit__(inout self, other: Device):
        self._type = other._type
        self._id = other._id

    fn __moveinit__(inout self, other: Device):
        self._type = other._type
        self._id = other._id

    fn __eq__(self, other: Device) -> Bool:
        return self._type == other._type and self._id == other._id

    fn __ne__(self, other: Device) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self._type == 0:
            return "cpu"
        elif self._type == 1:
            return "cuda:" + str(self._id)
        elif self._type == 2:
            return "metal:" + str(self._id)
        else:
            return "unknown"

    fn is_cpu(self) -> Bool:
        return self._type == 0

    fn is_cuda(self) -> Bool:
        return self._type == 1

    fn is_metal(self) -> Bool:
        return self._type == 2
