
struct DType(EqualityComparable):
    var _value: Int

    # We use static methods as factories instead of alias with heap types
    # or just simple integer constants wrapped in DType

    alias U8 = DType(0)
    alias U32 = DType(1)
    alias I16 = DType(2)
    alias I32 = DType(3)
    alias I64 = DType(4)
    alias BF16 = DType(5)
    alias F16 = DType(6)
    alias F32 = DType(7)
    alias F64 = DType(8)
    alias F8E4M3 = DType(9)
    alias F6E2M3 = DType(10)
    alias F6E3M2 = DType(11)
    alias F4 = DType(12)
    alias F8E8M0 = DType(13)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: DType) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: DType) -> Bool:
        return self._value != other._value

    fn as_str(self) -> String:
        if self == Self.U8: return "u8"
        if self == Self.U32: return "u32"
        if self == Self.I16: return "i16"
        if self == Self.I32: return "i32"
        if self == Self.I64: return "i64"
        if self == Self.BF16: return "bf16"
        if self == Self.F16: return "f16"
        if self == Self.F32: return "f32"
        if self == Self.F64: return "f64"
        if self == Self.F8E4M3: return "f8e4m3"
        if self == Self.F6E2M3: return "f6e2m3"
        if self == Self.F6E3M2: return "f6e3m2"
        if self == Self.F4: return "f4"
        if self == Self.F8E8M0: return "f8e8m0"
        return "unknown"

    fn size_in_bytes(self) -> Int:
        if self == Self.U8 or self == Self.F8E4M3 or self == Self.F8E8M0:
            return 1
        elif self == Self.I16 or self == Self.BF16 or self == Self.F16:
            return 2
        elif self == Self.U32 or self == Self.I32 or self == Self.F32:
            return 4
        elif self == Self.I64 or self == Self.F64:
            return 8
        else:
            return 0 # Compressed/sub-byte types

    fn is_int(self) -> Bool:
        return self == Self.U8 or self == Self.U32 or self == Self.I16 or self == Self.I32 or self == Self.I64

    fn is_float(self) -> Bool:
        return not self.is_int()
