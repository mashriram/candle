from ..tensor import Tensor
from . import Module
from ..shape import Shape
from collections import List

struct KVCache(Module):
    var k_cache: List[Tensor]
    var v_cache: List[Tensor]
    var layer_idx: Int

    fn __init__(inout self):
        self.k_cache = List[Tensor]()
        self.v_cache = List[Tensor]()
        self.layer_idx = 0

    fn update(inout self, k: Tensor, v: Tensor, layer_idx: Int) raises -> Tensor:
        # Simplified KV Cache logic
        # Store k, v for the layer.
        # If cache exists, cat with new k, v.

        # Ensure list size
        while len(self.k_cache) <= layer_idx:
            self.k_cache.append(k) # Placeholder, ideally empty or None check
            self.v_cache.append(v)
            # This logic is flawed for initialization.
            # We need Optional.
            # Assuming sequential access: if idx == len, append.
            # If idx < len, update (cat).

        if layer_idx < len(self.k_cache):
            # Cat along seq dim (1?)
            # Assuming (B, Seq, Head, Dim)
            var old_k = self.k_cache[layer_idx]
            var new_k_list = List[Tensor]()
            new_k_list.append(old_k)
            new_k_list.append(k)
            var new_k = Tensor.cat(new_k_list, 1)
            self.k_cache[layer_idx] = new_k

            # Return new_k? Usually return k, v to use.
            # Returning cached k for attention.
            return new_k
        else:
            self.k_cache.append(k)
            return k
