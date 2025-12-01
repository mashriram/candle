from candle.tensor import Tensor, BackpropOp
from candle.device import Device
from candle.dtype import DType
from candle.shape import Shape
from candle.nn.linear import Linear
from candle.optim import SGD
from collections import List
import math

fn main() raises:
    print("Running Candle Mojo Training Example...")
    var device = Device.CPU

    # 1. Setup Data: Simple Linear Regression y = 2x + 1
    # X: (4, 1)
    var x_data = List[Float32]()
    x_data.append(1.0); x_data.append(2.0); x_data.append(3.0); x_data.append(4.0)
    var X = Tensor.new(x_data, Shape(4, 1), device)

    # Y: (4, 1)
    var y_data = List[Float32]()
    y_data.append(3.0); y_data.append(5.0); y_data.append(7.0); y_data.append(9.0)
    var Y = Tensor.new(y_data, Shape(4, 1), device)

    # 2. Setup Model
    # Linear(1 -> 1)
    # Weight: (1, 1)
    var w_data = List[Float32]()
    w_data.append(0.5) # Initial guess
    var W = Tensor.from_storage(
        Storage(w_data, DType.F32, device),
        Layout.contiguous(Shape(1, 1)),
        BackpropOp.none(),
        True # is_variable
    )

    # Bias: (1,)
    var b_data = List[Float32]()
    b_data.append(0.0) # Initial guess
    var B = Tensor.from_storage(
        Storage(b_data, DType.F32, device),
        Layout.contiguous(Shape(1)),
        BackpropOp.none(),
        True # is_variable
    )

    var model = Linear(W, B)
    var optimizer = SGD(0.01) # Learning rate

    print("Initial Weight: " + str(model.weight))
    print("Initial Bias: " + str(model.bias))

    # 3. Training Loop
    for epoch in range(10):
        # Forward
        var pred = model.forward(X)

        # Loss (MSE) = (pred - Y)^2
        var diff = pred.sub(Y)
        var sqr_diff = diff.sqr()

        # Backward
        var grads_map = sqr_diff.backward()

        # Step (Functional update)
        var params = List[Tensor]()
        params.append(model.weight)
        params.append(model.bias)

        var new_params = optimizer.step_and_replace(params, grads_map.grads, grads_map.ids)

        # Update model
        model.weight = new_params[0]
        model.bias = new_params[1]

        if epoch % 2 == 0:
             # Basic loss logging (just inspecting data of first element for now)
             # Real implementation would have .item()
             print("Epoch " + str(epoch) + " | Weight: " + str(model.weight) + " | Bias: " + str(model.bias))

    print("Final Weight: " + str(model.weight))
    print("Final Bias: " + str(model.bias))
    print("Target: 2.0, 1.0")
