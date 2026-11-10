```toml
name = "DequantizeLinear"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the dequantizing dimension of the input tensor. Used for per-axis and blocked quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`."
default = 1

[attributes.block_size]
type = "INT"
desc = "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, `y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is `[ceil(Di/Si), ceil(Di/(Si-1))-1]`"
default = 0

[[inputs]]
name = "x"
type = "T1"
desc = "N-D quantized input tensor to be de-quantized."

[[inputs]]
name = "x_scale"
type = "T2"
desc = "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has the same shape as the input, except for one dimension in which blocking is performed."

[[inputs]]
name = "x_zero_point"
type = "T1"
desc = "Zero point for input `x`. Shape must match x_scale. It's optional. Zero point is 0 when it's not specified."
option = "optional"

[[outputs]]
name = "y"
type = "T2"
desc = "N-D full precision output tensor. It has same shape as input `x`."

[type_constraints.T1]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(int16)",
    "tensor(uint16)",
    "tensor(int32)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
    "tensor(uint4)",
    "tensor(int4)",
]
desc = "The type of the inputs 'x_zero_point' and 'x'."

[type_constraints.T2]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "'x_scale' determines the output type."
```

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.
`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 types quantization, but the dequantization formula remains the same
for consistency, and `x_scale` still determines the output type.
