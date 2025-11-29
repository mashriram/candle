from ..tensor import Tensor
from . import Module
from ..shape import Shape
from collections import List

struct Conv2d(Module):
    var weight: Tensor
    var bias: Tensor
    var stride: Int
    var padding: Int

    fn __init__(inout self, weight: Tensor, bias: Tensor, stride: Int, padding: Int):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

    fn forward(self, x: Tensor) raises -> Tensor:
        # x: (B, Cin, H, W)
        # weight: (Cout, Cin, K, K)
        # bias: (Cout,)

        var shape_x = x.shape().dims
        var shape_w = self.weight.shape().dims

        var B = shape_x[0]
        var Cin = shape_x[1]
        var H = shape_x[2]
        var W = shape_x[3]

        var Cout = shape_w[0]
        var K = shape_w[2]

        var H_out = (H + 2 * self.padding - K) // self.stride + 1
        var W_out = (W + 2 * self.padding - K) // self.stride + 1

        var out_data = List[Float32](capacity=B*Cout*H_out*W_out)

        # Flatten inputs for access
        var x_data = x.flatten_to_list()
        var w_data = self.weight.flatten_to_list()
        var b_data = self.bias.flatten_to_list()

        # Strides for access
        var x_stride_B = Cin * H * W
        var x_stride_C = H * W
        var x_stride_H = W

        var w_stride_Cout = Cin * K * K
        var w_stride_Cin = K * K
        var w_stride_K = K

        for b in range(B):
            for cout in range(Cout):
                var bias_val = b_data[cout]

                for h_out in range(H_out):
                    for w_out in range(W_out):

                        var h_start = h_out * self.stride - self.padding
                        var w_start = w_out * self.stride - self.padding

                        var sum_val: Float32 = 0.0

                        for cin in range(Cin):
                            for kh in range(K):
                                for kw in range(K):
                                    var h_in = h_start + kh
                                    var w_in = w_start + kw

                                    if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                                        var x_idx = b * x_stride_B + cin * x_stride_C + h_in * x_stride_H + w_in
                                        var w_idx = cout * w_stride_Cout + cin * w_stride_Cin + kh * w_stride_K + kw
                                        sum_val += x_data[x_idx] * w_data[w_idx]

                        out_data.append(sum_val + bias_val)

        var out_shape = Shape(B, Cout, H_out, W_out)
        # Note: This operation is not yet recorded in the backprop graph (BackpropOp.none()),
        # effectively acting as a "no_grad" block for Conv2d.
        # To enable training Conv2d, we need to register Op.conv2d and implement its backward.
        # Given the "Naive" implementation constraint, this forward pass satisfies correctness for inference.
        # Training Conv2d is a huge task (col2im), so we accept this limitation for this step.
        return Tensor.new(out_data, out_shape, x.device())
