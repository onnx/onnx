```toml
name = "DequantizeLinear"
domain = "ai.onnx"
shape_inference = "defs::quantization::utils::DequantizeLinearShapeInference_opset25"

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
    "tensor(uint2)",
    "tensor(int2)",
]
desc = "The type of the inputs 'x_zero_point' and 'x'."

[type_constraints.T2]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(float8e8m0)"]
desc = "The type of the input 'x_scale'."

[type_constraints.T3]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "The type of the output 'y'."
```
