```toml
name = "Dropout"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.is_test]
type = "INT"
desc = "(int, default 0) if nonzero, run dropout in test mode where the output is simply Y = X."
default = 0

[attributes.ratio]
type = "FLOAT"
desc = "(float, default 0.5) the ratio of random dropout"
default = 0.5
```
