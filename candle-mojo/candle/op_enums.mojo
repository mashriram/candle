
struct UnaryOp:
    var value: Int
    alias Exp = 0
    alias Log = 1
    alias Sin = 2
    alias Cos = 3
    alias Abs = 4
    alias Neg = 5
    alias Recip = 6
    alias Sqr = 7
    alias Sqrt = 8
    alias Relu = 9

struct BinaryOp:
    var value: Int
    alias Add = 0
    alias Sub = 1
    alias Mul = 2
    alias Div = 3
    alias Matmul = 4 # Matmul is often treated separately but fitting here for enum simplicity if needed
