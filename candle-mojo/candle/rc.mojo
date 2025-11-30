from memory import UnsafePointer

struct Rc[T: CollectionElement](CollectionElement):
    var _ptr: UnsafePointer[Int]

    fn __init__(inout self, value: T):
        var box_ptr = UnsafePointer[RcBox[T]].alloc(1)
        (box_ptr).init_pointee_move(RcBox[T](1, value))
        self._ptr = box_ptr.bitcast[Int]()

    fn __copyinit__(inout self, other: Rc[T]):
        self._ptr = other._ptr
        self._retain()

    fn __moveinit__(inout self, other: Rc[T]):
        self._ptr = other._ptr
        other._ptr = UnsafePointer[Int]()

    fn __del__(owned self):
        if not self._ptr:
            return
        self._release()

    fn _retain(self):
        if not self._ptr: return
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        box_ptr[].count += 1

    fn _release(self):
        if not self._ptr: return
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        box_ptr[].count -= 1
        if box_ptr[].count == 0:
            box_ptr.destroy_pointee()
            box_ptr.free()

    # Returns immutable copy (value semantics)
    fn __getitem__(self) -> T:
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        return box_ptr[].value

    # Returns mutable pointer to the contained value
    # Crucial for in-place updates (Optimizer)
    fn get_mut(self) -> UnsafePointer[T]:
        var box_ptr = self._ptr.bitcast[RcBox[T]]()
        return UnsafePointer.address_of(box_ptr[].value)

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
