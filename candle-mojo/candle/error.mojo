
struct Error(CollectionElement):
    var msg: String

    fn __init__(inout self, msg: String):
        self.msg = msg

    fn __copyinit__(inout self, other: Error):
        self.msg = other.msg

    fn __moveinit__(inout self, other: Error):
        self.msg = other.msg

    fn __str__(self) -> String:
        return self.msg

# In Mojo, we usually use the Error type or raises for Result-like behavior.
# We can define a helper for common errors.

fn bail(msg: String) raises:
    raise Error(msg)
