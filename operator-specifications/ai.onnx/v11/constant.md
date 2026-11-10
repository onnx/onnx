```toml
name = "Constant"
domain = "ai.onnx"

[attributes.sparse_value]
type = "SPARSE_TENSOR"
desc = "The value for the elements of the output tensor in sparse format."

[attributes.value]
type = "TENSOR"
desc = "The value for the elements of the output tensor."
```

A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
