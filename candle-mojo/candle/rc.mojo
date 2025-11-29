from memory import UnsafePointer

struct Rc[T: CollectionElement](CollectionElement):
    # Pointer to the managed block
    # Layout of block: [count: Int, data: T]
    var _ptr: UnsafePointer[Int]

    fn __init__(inout self, value: T):
        # Allocate memory for count (Int) and value (T)
        # We need size of Int + size of T
        # UnsafePointer is typed, so we can't easily mix types in one allocation without casting.
        # Strategy: Allocate 2 pointers. No, that's inefficient.
        # Strategy: Use a struct wrapper.

        # We cannot define a struct that wraps T easily in generic context without T being fixed?
        # Mojo generics are strong.
        # Let's allocate a pointer to a specific wrapper struct.
        # But `RcBox[T]` definition needs to be visible.

        # Simpler manual layout:
        # We allocate generic memory.
        # But let's use a Heap wrapper.

        # Since I can't easily define a private generic struct inside a function,
        # I'll rely on UnsafePointer.alloc(1) of a `RcBox[T]`.

        var box_ptr = UnsafePointer[RcBox[T]].alloc(1)
        # Initialize count to 1
        (box_ptr).init_pointee_move(RcBox[T](1, value))

        # Cast to Int pointer for internal storage just to be generic?
        # No, store the typed pointer.
        # But `Rc` needs to be `Rc[T]`.

        # Wait, if `Rc` is generic, `_ptr` can be `UnsafePointer[RcBox[T]]`.
        self._ptr = box_ptr.bitcast[Int]() # Erase type for now if needed, or keep typed.
        # Let's try keeping it typed but we need `RcBox` defined first.

    fn __copyinit__(inout self, other: Rc[T]):
        self._ptr = other._ptr
        self._retain()

    fn __moveinit__(inout self, other: Rc[T]):
        self._ptr = other._ptr
        # Nullify other to prevent decrement on other's del
        other._ptr = UnsafePointer[Int]()

    fn __del__(owned self):
        if not self._ptr:
            return
        self._release()

    fn _retain(self):
        if not self._ptr: return
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        # Increment count
        # In single threaded context:
        box_ptr[].count += 1

    fn _release(self):
        if not self._ptr: return
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        box_ptr[].count -= 1
        if box_ptr[].count == 0:
            # Destroy T
            box_ptr.destroy_pointee()
            # Free memory
            box_ptr.free()

    fn get_mut(self) -> UnsafePointer[T]:
        # Returns raw pointer to data
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        return UnsafePointer.address_of(box_ptr[].value)

    fn __getitem__(self) -> T:
        # Return copy of T? Or ref?
        # Structs in Mojo are values. This returns a copy.
        # To access reference, use `get_mut` or `__getitem__` returning ref (if supported).
        # For now, return copy is safe but maybe not what we want for "SharedStorage".
        # We want shared access.
        # If T is `Storage`, it holds `List`. `List` is a handle. Copying `Storage` copies the handle.
        # So copying `Storage` is cheap and CORRECT for shared access to the underlying List data!
        # So `__getitem__` returning `T` is fine for `Storage`.
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        return box_ptr[].value

struct RcBox[T: CollectionElement](CollectionElement):
    var count: Int
    var value: T

    fn __init__(inout self, count: Int, value: T):
        self.count = count
        self.value = value

    fn __copyinit__(inout self, other: RcBox[T]):
        self.count = other.count
        self.value = other.value

    fn __moveinit__(inout self, other: RcBox[T]):
        self.count = other.count
        self.value = other.value
