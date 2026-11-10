```toml
name = "CastLike"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::CastLikeShapeInference_opset19"
function_body_builders = { 19 = "defs::tensor::utils::CastLikeFunctionBody_opset19" }

[attributes.saturate]
type = "INT"
desc = "The parameter defines how the conversion behaves if an input value is out of range of the destination type. It only applies for float 8 conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. Please refer to operator Cast description for further details."
default = 1

[type_constraints.T1]
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
    "tensor(string)",
    "tensor(bfloat16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain input types. Casting from complex is not supported."

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
    "tensor(string)",
    "tensor(bfloat16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain output types. Casting to complex is not supported."
```
