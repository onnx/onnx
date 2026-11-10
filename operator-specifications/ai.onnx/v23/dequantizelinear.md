```toml
name = "DequantizeLinear"
domain = "ai.onnx"
shape_inference = "defs::quantization::utils::DequantizeLinearShapeInference_opset23"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the dequantizing dimension of the input tensor. Used for per-axis and blocked quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`."
default = 1

[attributes.block_size]
type = "INT"
desc = "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, `y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is `[ceil(Di/Si), ceil(Di/(Si-1))-1]`"
default = 0

[attributes.output_dtype]
type = "INT"
desc = "(Optional) The output data type. If not supplied, the output data type is inferred from `x_scale` data type (`T2`)"
default = 0

[[outputs]]
name = "y"
type = "T3"
desc = "N-D full precision output tensor. It has the same shape as input `x`. The data type is specified by the `output_dtype` attribute or, in its absence, the type of `x_scale`."

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
    "tensor(float4e2m1)",
]
desc = "The type of the inputs 'x_zero_point' and 'x'."

[type_constraints.T2]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "The type of the input 'x_scale'."

[type_constraints.T3]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "The type of the output 'y'."
```

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
is the same as `x_scale`. The output type also determines the precision of the multiplication operation.
