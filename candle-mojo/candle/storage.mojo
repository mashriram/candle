from collections import List
from .dtype import DType
from .device import Device

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
