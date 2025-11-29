from candle.tensor import Tensor
from candle.device import Device
from candle.dtype import DType
from candle.shape import Shape
from collections import List

fn main() raises:
    print("Running Candle Mojo Example...")

    var device = Device.CPU

    # 1. Tensor creation
    print("Creating tensors...")
    var data_a = List[Float32]()
    data_a.append(1.0)
    data_a.append(2.0)
    data_a.append(3.0)
    data_a.append(4.0)
    data_a.append(5.0)
    data_a.append(6.0)

    var shape_a = Shape(2, 3)
    var a = Tensor.new(data_a, shape_a, device)
    print("Tensor A created:")
    print(str(a))

    var data_b = List[Float32]()
    data_b.append(1.0)
    data_b.append(2.0)
    data_b.append(3.0)
    data_b.append(4.0)
    data_b.append(5.0)
    data_b.append(6.0)
    data_b.append(7.0)
    data_b.append(8.0)
    data_b.append(9.0)
    data_b.append(10.0)
    data_b.append(11.0)
    data_b.append(12.0)

    var shape_b = Shape(3, 4)
    var b = Tensor.new(data_b, shape_b, device)
    print("Tensor B created:")
    print(str(b))

    # 2. Matrix Multiplication
    print("Performing MatMul A @ B...")
    var c = a.matmul(b)
    print("Result C:")
    print(str(c))

    # 3. Addition
    print("Creating Tensor D (zeros)...")
    var shape_c = c.shape()
    var d = Tensor.zeros(shape_c, DType.F32, device)
    print(str(d))

    print("Adding C + D...")
    var e = c.add(d)
    print("Result E:")
    print(str(e))

    print("Done.")
