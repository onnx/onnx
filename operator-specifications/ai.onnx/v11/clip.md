```toml
name = "Clip"
domain = "ai.onnx"
attributes = {}
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor whose elements to be clipped"

[[inputs]]
name = "min"
type = "T"
desc = "Minimum value, under which element is replaced by min. It must be a scalar(tensor of empty shape)."
option = "optional"

[[inputs]]
name = "max"
type = "T"
desc = "Maximum value, above which element is replaced by max. It must be a scalar(tensor of empty shape)."
option = "optional"
```

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
