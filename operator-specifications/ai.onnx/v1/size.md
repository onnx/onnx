```toml
name = "Size"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SizeShapeInference_opset1"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."

[[outputs]]
name = "size"
type = "T1"
desc = "Total number of elements of the input tensor"

[type_constraints.T]
types = "all_tensor_types"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor, which should be a scalar though."
```

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
