from candle.tensor import Tensor
from collections import List

struct SGD:
    var learning_rate: Float32

    fn __init__(inout self, lr: Float32):
        self.learning_rate = lr

    fn step(self, params: List[Tensor], grads: List[Tensor], grad_ids: List[Int]) raises:
        # In Mojo, structs passed by value cannot be updated in-place to affect the caller
        # unless they wrap a reference type.
        # Since Tensor wraps Rc[Storage], we CAN update the storage content in place!
        # But `params` passed here are copies of the Tensor structs.
        # If we modify `params[i]._storage`, it modifies the shared data.
        #
        # HOWEVER, `Tensor` operations like `sub` currently return NEW Tensors with NEW Storage.
        # To support `step` (in-place), we need `sub_assign` or manual loop update on storage.
        #
        # Implementing manual loop update on Storage to support "true" step.

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
                # Update p._storage in place
                # p_data -= lr * g_data
                # We need access to storage pointer or list.
                # Rc gives us copy of Storage struct (which holds List handle).
                # List handle allows mutation.

                # We need to flatten g to match p's layout if contiguous.
                # Assuming params are contiguous weights.
                var p_data = p._storage[].data # Access List
                var g_data = g.flatten_to_list() # This creates copy.
                # We need g's data.

                # Check sizes
                if len(p_data) != len(g_data):
                    # Warning or skip
                    continue

                for k in range(len(p_data)):
                    p_data[k] = p_data[k] - self.learning_rate * g_data[k]

    fn step_and_replace(self, params: List[Tensor], grads: List[Tensor], grad_ids: List[Int]) raises -> List[Tensor]:
        # Legacy functional wrapper, now we can support in-place with shared storage logic above,
        # but let's keep this as fallback or utility.
        var new_params = List[Tensor]()
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
                var lr_data = List[Float32]()
                lr_data.append(self.learning_rate)
                var lr_t = Tensor.new(lr_data, Shape(1), p.device())
                var delta = g.mul(lr_t)
                var new_p = p.sub(delta)
                new_params.append(new_p)
            else:
                new_params.append(p)
        return new_params
