```toml
name = "QuantizeLinear"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the quantization dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."
default = 1

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
type = "T1"
desc = "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer quantization, or a 1-D Tensor for per-axis quantization."

[[inputs]]
name = "y_zero_point"
type = "T2"
desc = "Zero point for doing quantization to get 'y'. Shape must match y_scale. Default is uint8 with zero point of 0 if it's not specified."
option = "optional"

[type_constraints.T1]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"]
desc = "Constrain 'x' to float, float16, bfloat16 or int32 tensor."

[type_constraints.T2]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain 'y_zero_point' and 'y' to 8-bit integer/float tensor."
```

The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
'y_zero_point' and 'y' must have same type.
'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
but the quantization formula remains the same for consistency and
the type of the attribute 'y_zero_point' still determines the quantization type.
