```toml
name = "Clip"
domain = "ai.onnx"
node_determinism = "deterministic"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor whose elements to be clipped"
differentiation = "differentiable"

[[inputs]]
name = "min"
type = "T"
desc = "Minimum value, under which element is replaced by min. It must be a scalar(tensor of empty shape)."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "max"
type = "T"
desc = "Maximum value, above which element is replaced by max. It must be a scalar(tensor of empty shape)."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor with clipped input elements"
differentiation = "differentiable"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to all numeric tensors."
```

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
When 'min' is greater than 'max', the clip operator sets all the 'input' values to
the value of 'max'. Thus, this is equivalent to 'Min(max, Max(input, min))'.
