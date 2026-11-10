```toml
name = "BitwiseXor"
domain = "ai.onnx"
shape_inference = "defs::logical::utils::BinaryBitwiseShapeInference"

[[inputs]]
name = "A"
type = "T"
desc = "First input operand for the bitwise operator."
differentiation = "non_differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Second input operand for the bitwise operator."
differentiation = "non_differentiable"

[[outputs]]
name = "C"
type = "T"
desc = "Result tensor."
differentiation = "non_differentiable"

[type_constraints.T]
types = [
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
]
desc = "Constrain input to integer tensors."
```

Returns the tensor resulting from performing the bitwise `xor` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
