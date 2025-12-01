from ..tensor import Tensor
from ..error import Error

trait Module:
    fn forward(self, xs: Tensor) raises -> Tensor: ...
