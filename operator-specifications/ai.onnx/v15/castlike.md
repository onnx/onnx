```toml
name = "CastLike"
domain = "ai.onnx"
node_determinism = "deterministic"

[[inputs]]
name = "input"
type = "T1"
desc = "Input tensor to be cast."
differentiation = "differentiable"

[[inputs]]
name = "target_type"
type = "T2"
desc = "The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor produced by casting the first input tensor to have the same type as the second input tensor."
differentiation = "differentiable"

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
]
desc = "Constrain output types. Casting to complex is not supported."
```

The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
