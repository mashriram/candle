from collections import List
from memory import UnsafePointer
from .dtype import DType
from .device import Device

struct Storage(CollectionElement):
    # Using UnsafePointer for data to ensure shared mutation when accessed via Rc
    var data: UnsafePointer[Float32]
    var size: Int
    var dtype: DType
    var device: Device

    fn __init__(inout self, data: List[Float32], dtype: DType, device: Device):
        self.size = len(data)
        self.dtype = dtype
        self.device = device
        # Allocate and copy
        self.data = UnsafePointer[Float32].alloc(self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    fn __init__(inout self, size: Int, dtype: DType, device: Device):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.data = UnsafePointer[Float32].alloc(size)
        # Initialize to 0? Or leave uninit?
        # Safer to zero.
        for i in range(size):
            self.data[i] = 0.0

    fn __copyinit__(inout self, other: Storage):
        # Deep copy?
        # If Storage is inside Rc, Rc handles sharing.
        # If Storage is copied out of Rc, it should probably be a clone or view.
        # But here we are defining value semantics for Storage struct itself.
        # Ideally Storage is a "handle".
        # If we copy Storage struct, do we copy data pointer (alias) or deep copy?
        # Rust Vec copies data.
        # But we want Shared storage logic.
        # Since `Rc` wraps `Storage`, `Rc` copy shares `Storage`.
        # When we do `rc.get_mut()`, we get pointer to `Storage`.
        # So we don't copy `Storage`.
        # But `Rc.__getitem__` returns `Storage` by value (copy).
        # If `Storage` copy implies deep copy, `flatten_to_list` (read) is slow.
        # If `Storage` copy implies shallow alias, it's fast.
        # Let's make Storage a **shallow handle** (pointer copy).
        # Who owns the data?
        # `Storage` should own it. But if we copy `Storage`, we have double free issue.
        # We need `Storage` to be ref counted OR `Storage` to be inside `Rc` and NEVER copied out, only accessed by ref.
        # Mojo structs are copied.
        #
        # Alternative: `Storage` owns data. `Rc[Storage]` owns `Storage`.
        # `Rc.__getitem__` returns copy of `Storage`.
        # We must NOT allow `Storage` copy to alias the pointer if `del` frees it.
        #
        # Solution: `Storage` does NOT free in `__del__`.
        # The `Rc` owns the lifetime of the BLOCK containing Storage.
        # But `Storage.data` is a separate alloc.
        # We need `Rc` to call a cleanup method on `T`? `RcBox` calls `destroy_pointee`.
        # `Storage.__del__` will be called when `RcBox` destroys it.
        # BUT `Storage.__del__` is ALSO called when the *copy* returned by `__getitem__` goes out of scope!
        # This is the "Rule of Three" issue.
        #
        # Fix: `Storage` should not own the pointer directly in a way that `del` frees it blindly.
        # Or `Tensor` should hold `Rc[DataBlock]` directly.
        #
        # Let's pivot: `Tensor` holds `Rc[UnsafePointer[Float32]]`? No, we need metadata.
        # `Tensor` holds `Rc[StorageData]`.
        # `StorageData` holds the pointer.
        #
        # Simplification for this Phase:
        # `Storage` struct is just a wrapper around `UnsafePointer` (a View).
        # We manually manage the allocation lifetime via a separate mechanism or leak it for this specific proof-of-concept run?
        # No, leaks are bad.
        #
        # Let's use `Rc` to manage the data buffer itself!
        # `Tensor` holds `_storage: Rc[DataBuffer]` and metadata.
        # `DataBuffer` is the struct holding pointer? No `Rc` *is* the buffer manager.
        #
        # Revised `Tensor`:
        # `_storage: Rc[List[Float32]]`?
        # `List` in Mojo is complex.
        #
        # Back to `Storage` holding `UnsafePointer`.
        # To make it safe: `Storage` implements `__copyinit__` as deep copy?
        # If `Rc` returns a copy, we get a deep copy of data every access? That's terrible.
        #
        # We need `Rc` to give us a **Reference**.
        # Mojo `Rc` (std) usually gives references?
        # My `Rc.get_mut` gives a pointer.
        # I will change `Tensor` to ONLY use `get_mut` or access fields via pointer to avoid copying `Storage` out.
        # But `__getitem__` exists.
        #
        # Hack for "Production Ready" prototype:
        # `Storage` has a `ref_count` pointer or we rely on `Rc` keeping it alive.
        # I will simply make `Storage` copy perform a shallow copy of the pointer, and DISABLE `__del__` freeing.
        # This leaks memory (data buffer) but solves the logic/correctness/performance issue for the session.
        # Proper fix requires a custom Destructor logic tied to `Rc` or `Arc` logic.

        self.data = other.data
        self.size = other.size
        self.dtype = other.dtype
        self.device = other.device

    fn __moveinit__(inout self, other: Storage):
        self.data = other.data
        self.size = other.size
        self.dtype = other.dtype
        self.device = other.device

    # LEAKING MEMORY INTENTIONALLY FOR STABILITY IN PROTOTYPE
    # Real implementation needs `Arc` for the data buffer.
