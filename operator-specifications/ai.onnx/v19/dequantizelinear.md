```toml
name = "DequantizeLinear"
domain = "ai.onnx"
shape_inference = "defs::quantization::utils::DequantizeLinearShapeInference_opset19"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, rendering the axis unnecessary in this scenario."
default = 1

[[inputs]]
name = "x"
type = "T1"
desc = "N-D quantized input tensor to be de-quantized."

[[inputs]]
name = "x_scale"
type = "T2"
desc = "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, or a 1-D tensor for per-axis dequantization."

[[inputs]]
name = "x_zero_point"
type = "T1"
desc = "Zero point for input 'x'. Shape must match x_scale. It's optional. Zero point is 0 when it's not specified."
option = "optional"

[[outputs]]
name = "y"
type = "T2"
desc = "N-D full precision output tensor. It has same shape as input 'x'."

[type_constraints.T1]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(int32)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain 'x_zero_point' and 'x' to 8-bit integer or float, or /32-bit integer tensor."

[type_constraints.T2]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "'x_scale' determines the output type."
```

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
