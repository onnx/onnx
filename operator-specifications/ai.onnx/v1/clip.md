```toml
name = "Clip"
domain = "ai.onnx"
shape_inference = false

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[attributes.max]
type = "FLOAT"
desc = "Maximum value, above which element is replaced by max"

[attributes.min]
type = "FLOAT"
desc = "Minimum value, under which element is replaced by min"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor whose elements to be clipped"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor with clipped input elements"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
