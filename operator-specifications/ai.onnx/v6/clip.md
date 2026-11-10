```toml
name = "Clip"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.max]
type = "FLOAT"
desc = "Maximum value, above which element is replaced by max"
default = 3.4028235e+38

[attributes.min]
type = "FLOAT"
desc = "Minimum value, under which element is replaced by min"
default = -3.4028235e+38
```
