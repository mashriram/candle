from candle.tensor import Tensor
from candle.device import Device
from candle.dtype import DType
from candle.shape import Shape
from candle.nn.linear import Linear
from collections import List

fn main() raises:
    print("Running Candle Mojo Example...")
    var device = Device.CPU

    # 1. Advanced Ops: Broadcasting & Transpose
    print("\n--- Testing Broadcasting & Transpose ---")
    var shape_a = Shape(2, 3)
    var data_a = List[Float32]()
    # [1, 2, 3]
    # [4, 5, 6]
    for i in range(6): data_a.append(Float32(i+1))
    var A = Tensor.new(data_a, shape_a, device)

    # Broadcast add: (2, 3) + (1, 3)
    var shape_b = Shape(1, 3)
    var data_b = List[Float32]()
    # [10, 20, 30]
    data_b.append(10.0); data_b.append(20.0); data_b.append(30.0)
    var B = Tensor.new(data_b, shape_b, device)

    print("A:\n" + str(A))
    print("B:\n" + str(B))

    var C = A.add(B)
    print("A + B (Broadcasting):\n" + str(C))
    # Expected:
    # [11, 22, 33]
    # [14, 25, 36]

    print("A Transposed:\n" + str(A.transpose(0, 1)))

    # 2. Testing Linear Layer
    print("\n--- Testing Linear Layer ---")
    # Weights: (Out, In) -> (2, 3)
    var shape_w = Shape(2, 3)
    var data_w = List[Float32]()
    # w = [[1, 1, 1], [1, 1, 1]]
    for _ in range(6): data_w.append(1.0)
    var W = Tensor.new(data_w, shape_w, device)

    # Bias: (Out,) -> (2,) or (1, 2) for broadcasting?
    # Usually bias is (Out,), broadcasted to (Batch, Out).
    # Matmul result will be (Batch, Out).
    var shape_bias = Shape(2)
    var data_bias = List[Float32]()
    data_bias.append(0.5); data_bias.append(0.5)
    var Bias = Tensor.new(data_bias, shape_bias, device)

    var linear = Linear(W, Bias)

    # Input: (Batch, In) -> (1, 3)
    # x = [[1, 2, 3]]
    var shape_x = Shape(1, 3)
    var data_x = List[Float32]()
    data_x.append(1.0); data_x.append(2.0); data_x.append(3.0)
    var X = Tensor.new(data_x, shape_x, device)

    print("Input X:\n" + str(X))
    var output = linear.forward(X)
    print("Linear(X):\n" + str(output))
    # Calculation:
    # X @ W.T
    # [1, 2, 3] @ [[1, 1], [1, 1], [1, 1]] = [6, 6]
    # + Bias [0.5, 0.5]
    # = [6.5, 6.5]

    # 3. Unary Ops
    print("\n--- Testing Unary Ops ---")
    var neg_x = X.neg()
    print("Neg(X):\n" + str(neg_x))

    var sqr_x = X.sqr()
    print("Sqr(X):\n" + str(sqr_x))

    print("Done.")
