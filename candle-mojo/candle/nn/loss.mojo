from ..tensor import Tensor
from ..op_enums import BinaryOp
import math

fn mse_loss(pred: Tensor, target: Tensor) raises -> Tensor:
    var diff = pred.sub(target)
    var sqr = diff.sqr()
    # Mean over all elements
    # Currently sum/mean take dim.
    # Flatten first?
    var flat_sqr = sqr.reshape(Shape(sqr.shape().elem_count()))
    return flat_sqr.mean(0)

fn cross_entropy(pred: Tensor, target: Tensor) raises -> Tensor:
    # pred: logits (Batch, Classes)
    # target: class indices (Batch,) or probabilities (Batch, Classes)
    # Assuming indices for standard CE?
    # Or simplified: target is one-hot or same shape.
    # Let's assume target is same shape (probabilities) or we need index_select.
    # We implemented gather/index_select? No.
    # So assuming target is probabilities/soft-labels for now.

    # - sum(target * log_softmax(pred))
    var exps = pred.exp()
    var sum_exps = exps.sum(1)
    # Reshape sum
    var s_shape = sum_exps.shape().dims
    var new_dims = List[Int]()
    new_dims.append(s_shape[0])
    new_dims.append(1)
    var sum_r = sum_exps.reshape(Shape(new_dims))

    var log_probs = exps.div(sum_r).log()
    var loss = target.mul(log_probs).sum(1).mean(0).neg()
    return loss
