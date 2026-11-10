```toml
name = "QuantizeLinear"
domain = "ai.onnx"

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
desc = "Zero point for doing quantization to get `y`. Shape must match `y_scale`. Default is uint8 with zero point of 0 if it's not specified."
option = "optional"

[type_constraints.T1]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"]
desc = "The type of the input 'x'."

[type_constraints.T2]
types = [
    "tensor(float)",
    "tensor(float16)",
    "tensor(bfloat16)",
    "tensor(int32)",
    "tensor(float8e8m0)",
]
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
