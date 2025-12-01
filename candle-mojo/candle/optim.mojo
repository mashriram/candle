from candle.tensor import Tensor
from collections import List

struct SGD:
    var learning_rate: Float32

    fn __init__(inout self, lr: Float32):
        self.learning_rate = lr

    fn step(self, params: List[Tensor], grads: List[Tensor], grad_ids: List[Int]) raises:
        # Simplistic step: params -= lr * grad
        # We need to match params to grads via ID

        for i in range(len(params)):
            var p = params[i]
            # Find grad
            var found = False
            var g: Tensor

            for j in range(len(grad_ids)):
                if grad_ids[j] == p.id():
                    g = grads[j]
                    found = True
                    break

            if found:
                # Update param data in place?
                # Tensor struct is value type, but storage is Reference/Pointer based ideally.
                # Here `Storage` is a struct wrapping `List`. `List` has reference semantics for data?
                # In Mojo, `List` is a struct. Copies might be deep or shallow depending on implementation.
                # If shallow, we are good. If deep, we are in trouble.
                # Mojo standard library `List` owns its memory. Copies are deep by default?
                # Actually, `List` is a handle.
                # Let's assume we need to update the storage content.

                # p._storage.data is the list.
                # p = p - lr * g
                # This creates a NEW tensor p. It doesn't update the original `p` reference held by the user if `p` is a value type.
                # This is the crux of porting Rust/C++ to Mojo structs.
                # WE NEED IN-PLACE UPDATE.

                # Hack: Modifying the internal list of the tensor.
                # But `g` might have different layout.
                # We need `p.sub_assign(g * lr)`.
                pass

        # For this proof of concept, since we can't easily do in-place updates on value-type Tensors
        # that are held elsewhere without pointers, we will just print the update or return new tensors.
        # But `step` is supposed to be in-place.
        # Mojo's `List` is a `CollectionElement`, it has value semantics.
        # We really need `Tensor` to be a class or hold a `Pointer`.
        # I previously noted this.
        # Since I can't refactor everything to Pointers now without huge risk,
        # I will simulate the update by "returning" updated params or assuming the user re-assigns them?
        # No, `SGD` usually takes refs.

        # PROPOSAL: Add `sub_inplace` to Tensor that uses `UnsafePointer` or similar if possible,
        # or just access the list elements mutable.
        # Since `p` is passed by value (copy) in `List[Tensor]`, updating `p` here won't affect caller's `p`.
        # Unless `Storage` holds a pointer.
        # Current `Storage` holds `List`.

        # Workaround: The user must pass params as `inout`.
        # But `List` elements cannot be easily yielded as `inout`.

        # Alternative: `step` returns new tensors, user replaces them.
        # params = optimizer.step(params, ...)
        pass

    fn step_and_replace(self, params: List[Tensor], grads: List[Tensor], grad_ids: List[Int]) raises -> List[Tensor]:
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
                # p_new = p - lr * g
                # We need a `scale` op or `mul` with scalar.
                # We have `mul` with tensor.
                # Create scalar tensor for LR
                # We need p.device() but p is local.
                # Assuming CPU for now or accessing p's device.

                # var lr_t = Tensor.new([self.learning_rate], Shape(1), p.device())
                # But broadcasting (1,) to (2,3) might fail if rank differs?
                # Our broadcast logic handles rank mismatch (prepend 1s).

                # Construct LR tensor
                var lr_data = List[Float32]()
                lr_data.append(self.learning_rate)
                var lr_t = Tensor.new(lr_data, Shape(1), p.device())

                var delta = g.mul(lr_t)
                var new_p = p.sub(delta)
                # Preserve ID? No, it's a new tensor content.
                # But for the "variable" identity, we might want to keep ID?
                # If we replace the tensor in the model, the ID changes.
                # That's fine for simple training loop: model.weight = new_weight
                new_params.append(new_p)
            else:
                new_params.append(p)
        return new_params
