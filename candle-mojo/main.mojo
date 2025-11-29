from candle.tensor import Tensor, BackpropOp
from candle.device import Device
from candle.dtype import DType
from candle.shape import Shape
from candle.nn.linear import Linear
from candle.nn.layer_norm import LayerNorm
from candle.nn.activation import Activation
from candle.nn.loss import mse_loss
from candle.optim import SGD
from candle.rc import Rc
from candle.storage import Storage
from candle.layout import Layout
from collections import List
import math

fn main() raises:
    print("Running Candle Mojo Training Example (Deep Network)...")
    var device = Device.CPU

    # 1. Setup Data
    var x_data = List[Float32]()
    x_data.append(1.0); x_data.append(2.0); x_data.append(3.0); x_data.append(4.0)
    var X = Tensor.new(x_data, Shape(4, 1), device)

    var y_data = List[Float32]()
    y_data.append(3.0); y_data.append(5.0); y_data.append(7.0); y_data.append(9.0)
    var Y = Tensor.new(y_data, Shape(4, 1), device)

    # 2. Setup Model: Linear -> ReLU -> Linear
    # L1: 1 -> 4
    var w1_data = List[Float32]()
    for _ in range(4): w1_data.append(0.1)
    var W1 = Tensor.from_storage(Rc[Storage](Storage(w1_data, DType.F32, device)), Layout.contiguous(Shape(4, 1)), BackpropOp.none(), True)
    var b1_data = List[Float32]()
    for _ in range(4): b1_data.append(0.0)
    var B1 = Tensor.from_storage(Rc[Storage](Storage(b1_data, DType.F32, device)), Layout.contiguous(Shape(4)), BackpropOp.none(), True)

    var l1 = Linear(W1, B1)
    var relu = Activation.relu()

    # L2: 4 -> 1
    var w2_data = List[Float32]()
    for _ in range(4): w2_data.append(0.1)
    var W2 = Tensor.from_storage(Rc[Storage](Storage(w2_data, DType.F32, device)), Layout.contiguous(Shape(1, 4)), BackpropOp.none(), True)
    var b2_data = List[Float32]()
    b2_data.append(0.0)
    var B2 = Tensor.from_storage(Rc[Storage](Storage(b2_data, DType.F32, device)), Layout.contiguous(Shape(1)), BackpropOp.none(), True)

    var l2 = Linear(W2, B2)

    var optimizer = SGD(0.01)

    # 3. Training Loop
    for epoch in range(5):
        # Forward
        var x1 = l1.forward(X)
        var x2 = relu.forward(x1)
        var pred = l2.forward(x2)

        # Loss
        var loss = mse_loss(pred, Y)

        # Backward
        var grads_map = loss.backward()

        # Step
        # Using functional step_and_replace still, but we could use step() now if we passed params.
        # But main structure is fine.
        var params = List[Tensor]()
        params.append(l1.weight); params.append(l1.bias)
        params.append(l2.weight); params.append(l2.bias)

        # NOTE: optimizer.step(params, ...) works in-place on shared storage now.
        # But we need to pass params correctly.
        optimizer.step(params, grads_map.grads, grads_map.ids)

        # No need to re-assign model weights if step worked in-place on shared storage!
        # Verification:

        print("Epoch " + str(epoch) + " | Loss computed | L1 Weight[0]: " + str(l1.weight.flatten_to_list()[0]))

    print("Training finished.")
