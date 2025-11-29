# candle-mojo

A port of the [Candle](https://github.com/huggingface/candle) machine learning framework to Mojo.

**Note**: This is an early stage port focusing on the core tensor library (`candle-core`).

## Status

Currently implementing the foundational components:
- `DType`
- `Shape`
- `Device` (CPU only)
- `Tensor` (Basic structure and operations)

## Usage

(See `main.mojo` for a runnable example)

```mojo
from candle.tensor import Tensor
from candle.device import Device
from candle.dtype import DType

fn main():
    # Example usage
    pass
```
