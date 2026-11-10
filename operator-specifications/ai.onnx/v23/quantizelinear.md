```toml
name = "QuantizeLinear"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis and blocked quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, rendering the axis unnecessary in this scenario."
default = 1

[attributes.block_size]
type = "INT"
desc = "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, `y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is `[ceil(Di/Si), ceil(Di/(Si-1))-1]`"
default = 0

[attributes.output_dtype]
type = "INT"
desc = "(Optional) The output data type. If not supplied, the output data type is inferred from `y_zero_point` data type (`T3`). If neither `output_dtype` nor `y_zero_point` are supplied, output data type is uint8. If both `output_dtype` and `y_zero_point` are specified, `output_dtype` must be `T3`."
default = 0

[attributes.precision]
type = "INT"
desc = "(Optional) The precision of the division operation between `x` and `y_scale`. If not provided, it will be the same as the type of `y_scale`."
default = 0

[attributes.saturate]
type = "INT"
desc = "The parameter defines how the conversion behaves if an input value is out of range of the destination type. It only applies for float 8 quantization (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. All cases are fully described in two tables inserted in the operator description."
default = 1

[[inputs]]
name = "x"
type = "T1"
desc = "N-D full precision Input tensor to be quantized."

[[inputs]]
name = "y_scale"
type = "T2"
desc = "Scale for doing quantization to get `y`. For per-tensor/layer quantization the scale is a scalar, for per-axis quantization it is a 1-D Tensor and for blocked quantization it has the same shape as the input, except for one dimension in which blocking is performed."

[[inputs]]
name = "y_zero_point"
type = "T3"
desc = "Zero point for doing quantization to get `y`. Shape must match `y_scale`.Default is uint8 with zero point of 0 if it's not specified."
option = "optional"

[[outputs]]
name = "y"
type = "T3"
desc = "N-D quantized output tensor. It has same shape as input `x`."

[type_constraints.T1]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"]
desc = "The type of the input 'x'."

[type_constraints.T2]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"]
desc = "The type of the input 'y_scale'."

[type_constraints.T3]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(int16)",
    "tensor(uint16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
    "tensor(uint4)",
    "tensor(int4)",
    "tensor(float4e2m1)",
]
desc = "The type of the input `y_zero_point` and the output `y`."
```

The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
`x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
`y_scale`, unless the `precision` attribute is specified.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
 `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
