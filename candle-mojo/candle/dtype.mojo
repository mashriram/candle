
struct DType:
    var _value: Int
    var _name: String

    alias U8 = DType(0, "u8")
    alias U32 = DType(1, "u32")
    alias I16 = DType(2, "i16")
    alias I32 = DType(3, "i32")
    alias I64 = DType(4, "i64")
    alias BF16 = DType(5, "bf16")
    alias F16 = DType(6, "f16")
    alias F32 = DType(7, "f32")
    alias F64 = DType(8, "f64")
    alias F8E4M3 = DType(9, "f8e4m3")
    alias F6E2M3 = DType(10, "f6e2m3")
    alias F6E3M2 = DType(11, "f6e3m2")
    alias F4 = DType(12, "f4")
    alias F8E8M0 = DType(13, "f8e8m0")

    fn __eq__(self, other: DType) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: DType) -> Bool:
        return self._value != other._value

    fn as_str(self) -> String:
        return self._name

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
