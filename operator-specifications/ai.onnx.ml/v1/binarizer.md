```toml
name = "Binarizer"
domain = "ai.onnx.ml"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.threshold]
type = "FLOAT"
desc = "Values greater than this are mapped to 1, others to 0."
default = 0.0

[[inputs]]
name = "X"
type = "T"
desc = "Data to be binarized"

[[outputs]]
name = "Y"
type = "T"
desc = "Binarized output data"

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input must be a tensor of a numeric type. The output will be of the same tensor type."
```

Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
