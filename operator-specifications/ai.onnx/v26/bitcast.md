```toml
name = "BitCast"
domain = "ai.onnx"
node_determinism = "deterministic"

[attributes.to]
type = "INT"
desc = "The data type to which the input tensor is bitwise reinterpreted. Must be one of the non-string types from DataType enum in TensorProto. The target type must have the same bit-width as the input type."
required = true

[[inputs]]
name = "input"
type = "T1"
desc = "Input tensor to be bitcast."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor with the same shape as the input."
differentiation = "non_differentiable"

[type_constraints.T1]
types = "all_non_string_tensor_types_ir13"
desc = "Constrain input types. Bitcasting from string is not supported."

[type_constraints.T2]
types = "all_non_string_tensor_types_ir13"
desc = "Constrain output types. Bitcasting to string is not supported."
```

Reinterprets the binary representation of a tensor as a different data type,
specified by the 'to' attribute. Unlike Cast, BitCast preserves the exact bit
pattern without any value conversion.

The target data type must have the same bit-width as the input data type.
The output tensor has the same shape as the input tensor.
All types except string are supported. Implementations must treat the
underlying bytes as little endian.
