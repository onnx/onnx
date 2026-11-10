```toml
name = "Constant"
domain = "ai.onnx"

[attributes.sparse_value]
type = "SPARSE_TENSOR"
desc = "The value for the elements of the output tensor in sparse format."

[attributes.value]
type = "TENSOR"
desc = "The value for the elements of the output tensor."

[attributes.value_float]
type = "FLOAT"
desc = "The value for the sole element for the scalar, float32, output tensor."

[attributes.value_floats]
type = "FLOATS"
desc = "The values for the elements for the 1D, float32, output tensor."

[attributes.value_int]
type = "INT"
desc = "The value for the sole element for the scalar, int64, output tensor."

[attributes.value_ints]
type = "INTS"
desc = "The values for the elements for the 1D, int64, output tensor."

[attributes.value_string]
type = "STRING"
desc = "The value for the sole element for the scalar, UTF-8 string, output tensor."

[attributes.value_strings]
type = "STRINGS"
desc = "The values for the elements for the 1D, UTF-8 string, output tensor."
```

This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
