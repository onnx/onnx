```toml
name = "ConstantOfShape"
domain = "ai.onnx"
shape_inference = "defs::generator::utils::ConstantOfShapeShapeInference_opset20"

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain input types."

[type_constraints.T2]
types = [
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(bool)",
    "tensor(bfloat16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain output types to be numerics."
```
