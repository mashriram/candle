from candle.tensor import Tensor
from collections import List
from candle.shape import Shape
import math

struct SGD:
    var learning_rate: Float32

    fn __init__(inout self, lr: Float32):
        self.learning_rate = lr

    fn step(self, params: List[Tensor], grads: List[Tensor], grad_ids: List[Int]) raises:
        for i in range(len(params)):
            var p = params[i]
            var found = False
            var g: Tensor
            for j in range(len(grad_ids)):
                if grad_ids[j] == p.id():
                    g = grads[j]
                    found = True
                    break

            if found:
                var p_data = p._storage[].data
                var g_data = g.flatten_to_list()
                if len(p_data) != len(g_data): continue # Should warn
                for k in range(len(p_data)):
                    p_data[k] = p_data[k] - self.learning_rate * g_data[k]

struct AdamW:
    var learning_rate: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32

    # State tracking: m and v for each param ID.
    # Since we can't easily map ID -> Tensor in struct fields without Dict,
    # We will assume a fixed order of params or require state passing?
    # Or just use parallel lists.
    var m_states: List[Tensor]
    var v_states: List[Tensor]
    var param_ids: List[Int]
    var t: Int

    fn __init__(inout self, lr: Float32, beta1: Float32 = 0.9, beta2: Float32 = 0.999, eps: Float32 = 1e-8, weight_decay: Float32 = 0.01):
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m_states = List[Tensor]()
        self.v_states = List[Tensor]()
        self.param_ids = List[Int]()
        self.t = 0

    fn step(self, params: List[Tensor], grads: List[Tensor], grad_ids: List[Int]) raises:
        self.t += 1
        for i in range(len(params)):
            var p = params[i]
            var id = p.id()

            # Find or init state
            var state_idx = -1
            for k in range(len(self.param_ids)):
                if self.param_ids[k] == id:
                    state_idx = k
                    break

            if state_idx == -1:
                self.param_ids.append(id)
                self.m_states.append(p.zeros_like())
                self.v_states.append(p.zeros_like())
                state_idx = len(self.param_ids) - 1

            var m = self.m_states[state_idx]
            var v = self.v_states[state_idx]

            # Find grad
            var found = False
            var g: Tensor
            for j in range(len(grad_ids)):
                if grad_ids[j] == id:
                    g = grads[j]
                    found = True
                    break

            if found:
                var p_ptr = p._storage[].data
                var g_ptr = g.flatten_to_list() # Copy
                var m_ptr = m._storage[].data
                var v_ptr = v._storage[].data

                # Update loop
                for k in range(len(p_ptr)):
                    var grad_val = g_ptr[k] + self.weight_decay * p_ptr[k]

                    m_ptr[k] = self.beta1 * m_ptr[k] + (1.0 - self.beta1) * grad_val
                    v_ptr[k] = self.beta2 * v_ptr[k] + (1.0 - self.beta2) * grad_val * grad_val

                    var m_hat = m_ptr[k] / (1.0 - math.pow(self.beta1, self.t))
                    var v_hat = v_ptr[k] / (1.0 - math.pow(self.beta2, self.t))

                    p_ptr[k] = p_ptr[k] - self.learning_rate * m_hat / (math.sqrt(v_hat) + self.eps)
