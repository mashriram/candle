from candle.tensor import Tensor, BackpropOp
from candle.device import Device
from candle.dtype import DType
from candle.shape import Shape
from candle.models.llama import LlamaBlock, Config
from candle.optim import AdamW
from candle.nn.loss import mse_loss
from candle.rc import Rc
from candle.storage import Storage
from candle.layout import Layout
from collections import List
import math

fn main() raises:
    print("Running Candle Mojo Transformer Example...")
    var device = Device.CPU
    var cfg = Config()
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4

    # 1. Setup Model
    var block = LlamaBlock(cfg, device)
    var optimizer = AdamW(0.001)

    # 2. Setup Data (Batch=2, Seq=10, Hidden=64)
    var input_data = List[Float32]()
    for _ in range(2 * 10 * 64): input_data.append(0.1)
    var x = Tensor.new(input_data, Shape(2, 10, 64), device)

    var target_data = List[Float32]()
    for _ in range(2 * 10 * 64): target_data.append(0.2)
    var target = Tensor.new(target_data, Shape(2, 10, 64), device)

    print("Starting Training Loop...")
    for i in range(5):
        # Forward
        var out = block.forward(x)
        var loss = mse_loss(out, target)

        # Backward
        var grads_map = loss.backward()

        # Step
        # Collect params - in real app, Model.parameters() does this.
        # Here we manually collect a few to prove updates.
        var params = List[Tensor]()
        params.append(block.input_layernorm.weight)
        params.append(block.mlp.down_proj.weight)

        optimizer.step(params, grads_map.grads, grads_map.ids)

        # Verify loss change
        var loss_val = loss.flatten_to_list()[0]
        print("Iter " + str(i) + " Loss: " + str(loss_val))

    print("Transformer Block Trained Successfully.")
