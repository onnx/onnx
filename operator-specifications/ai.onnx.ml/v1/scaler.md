```toml
name = "Scaler"
domain = "ai.onnx.ml"
shape_inference = false

[attributes.offset]
type = "FLOATS"
desc = "First, offset by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count."

[attributes.scale]
type = "FLOATS"
desc = "Second, multiply by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.<br>Must be same length as 'offset'"

[[inputs]]
name = "X"
type = "T"
desc = "Data to be scaled."

[[outputs]]
name = "Y"
type = "tensor(float)"
desc = "Scaled output data."

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input must be a tensor of a numeric type."
```

Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
