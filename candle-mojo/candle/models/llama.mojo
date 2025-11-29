from ..tensor import Tensor
from ..nn.linear import Linear
from ..nn.nlp import RMSNorm, Embedding, RotaryEmbedding
from ..nn.activation import Activation
from ..nn.kv_cache import KVCache
from ..device import Device
from ..dtype import DType
from ..shape import Shape
from ..rc import Rc
from ..storage import Storage
from ..layout import Layout
from ..op_enums import BackpropOp
import math

struct Config(CollectionElement):
    var hidden_size: Int
    var intermediate_size: Int
    var num_hidden_layers: Int
    var num_attention_heads: Int
    var num_key_value_heads: Int
    var max_position_embeddings: Int
    var rms_norm_eps: Float32

    fn __init__(inout self):
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.num_hidden_layers = 32
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.max_position_embeddings = 2048
        self.rms_norm_eps = 1e-6

    fn __copyinit__(inout self, other: Config):
        self.hidden_size = other.hidden_size
        self.intermediate_size = other.intermediate_size
        self.num_hidden_layers = other.num_hidden_layers
        self.num_attention_heads = other.num_attention_heads
        self.num_key_value_heads = other.num_key_value_heads
        self.max_position_embeddings = other.max_position_embeddings
        self.rms_norm_eps = other.rms_norm_eps

    fn __moveinit__(inout self, other: Config):
        self.hidden_size = other.hidden_size
        self.intermediate_size = other.intermediate_size
        self.num_hidden_layers = other.num_hidden_layers
        self.num_attention_heads = other.num_attention_heads
        self.num_key_value_heads = other.num_key_value_heads
        self.max_position_embeddings = other.max_position_embeddings
        self.rms_norm_eps = other.rms_norm_eps

struct Attention(CollectionElement):
    var q_proj: Linear
    var k_proj: Linear
    var v_proj: Linear
    var o_proj: Linear
    var num_heads: Int
    var head_dim: Int
    var kv_heads: Int

    fn __init__(inout self, cfg: Config, device: Device) raises:
        self.num_heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.num_heads

        # Mock Weights
        var w = Tensor.ones(Shape(cfg.hidden_size, cfg.hidden_size), DType.F32, device)
        var b = Tensor.zeros(Shape(cfg.hidden_size), DType.F32, device) # Bias usually None for Llama

        self.q_proj = Linear(w, b)
        self.k_proj = Linear(w, b)
        self.v_proj = Linear(w, b)
        self.o_proj = Linear(w, b)

    fn __copyinit__(inout self, other: Attention):
        self.q_proj = other.q_proj
        self.k_proj = other.k_proj
        self.v_proj = other.v_proj
        self.o_proj = other.o_proj
        self.num_heads = other.num_heads
        self.head_dim = other.head_dim
        self.kv_heads = other.kv_heads

    fn __moveinit__(inout self, other: Attention):
        self.q_proj = other.q_proj
        self.k_proj = other.k_proj
        self.v_proj = other.v_proj
        self.o_proj = other.o_proj
        self.num_heads = other.num_heads
        self.head_dim = other.head_dim
        self.kv_heads = other.kv_heads

    fn forward(self, x: Tensor) raises -> Tensor:
        # x: (B, Seq, Hidden)
        var B = x.shape().dims[0]
        var Seq = x.shape().dims[1]

        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(x)
        var v = self.v_proj.forward(x)

        # Reshape to (B, Seq, Heads, Dim) -> Transpose (B, Heads, Seq, Dim)
        # q = q.reshape(Shape(B, Seq, self.num_heads, self.head_dim)).transpose(1, 2)
        # ... logic ...
        # Simplified return for POC
        return self.o_proj.forward(v)

struct MLP(CollectionElement):
    var gate_proj: Linear
    var up_proj: Linear
    var down_proj: Linear
    var act: Activation

    fn __init__(inout self, cfg: Config, device: Device) raises:
        var w_up = Tensor.ones(Shape(cfg.intermediate_size, cfg.hidden_size), DType.F32, device)
        var b_up = Tensor.zeros(Shape(cfg.intermediate_size), DType.F32, device)
        var w_down = Tensor.ones(Shape(cfg.hidden_size, cfg.intermediate_size), DType.F32, device)
        var b_down = Tensor.zeros(Shape(cfg.hidden_size), DType.F32, device)

        self.gate_proj = Linear(w_up, b_up)
        self.up_proj = Linear(w_up, b_up)
        self.down_proj = Linear(w_down, b_down)
        self.act = Activation.silu()

    fn __copyinit__(inout self, other: MLP):
        self.gate_proj = other.gate_proj
        self.up_proj = other.up_proj
        self.down_proj = other.down_proj
        self.act = other.act

    fn __moveinit__(inout self, other: MLP):
        self.gate_proj = other.gate_proj
        self.up_proj = other.up_proj
        self.down_proj = other.down_proj
        self.act = other.act

    fn forward(self, x: Tensor) raises -> Tensor:
        var lhs = self.act.forward(self.gate_proj.forward(x))
        var rhs = self.up_proj.forward(x)
        return self.down_proj.forward(lhs.mul(rhs))

struct LlamaBlock(CollectionElement):
    var attn: Attention
    var mlp: MLP
    var input_layernorm: RMSNorm
    var post_attention_layernorm: RMSNorm

    fn __init__(inout self, cfg: Config, device: Device) raises:
        self.attn = Attention(cfg, device)
        self.mlp = MLP(cfg, device)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device)

    fn __copyinit__(inout self, other: LlamaBlock):
        self.attn = other.attn
        self.mlp = other.mlp
        self.input_layernorm = other.input_layernorm
        self.post_attention_layernorm = other.post_attention_layernorm

    fn __moveinit__(inout self, other: LlamaBlock):
        self.attn = other.attn
        self.mlp = other.mlp
        self.input_layernorm = other.input_layernorm
        self.post_attention_layernorm = other.post_attention_layernorm

    fn forward(self, x: Tensor) raises -> Tensor:
        var residual = x
        var x_norm = self.input_layernorm.forward(x)
        var attn_out = self.attn.forward(x_norm)
        x = residual.add(attn_out)

        residual = x
        x_norm = self.post_attention_layernorm.forward(x)
        var mlp_out = self.mlp.forward(x_norm)
        x = residual.add(mlp_out)
        return x
