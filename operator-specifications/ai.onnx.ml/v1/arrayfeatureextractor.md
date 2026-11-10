```toml
name = "ArrayFeatureExtractor"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::ArrayFeatureExtractorShapeInference_opset1"

[[inputs]]
name = "X"
type = "T"
desc = "Data to be selected"

[[inputs]]
name = "Y"
type = "tensor(int64)"
desc = "The indices, based on 0 as the first index of any dimension."

[[outputs]]
name = "Z"
type = "T"
desc = "Selected output data as an array"

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)", "tensor(string)"]
desc = "The input must be a tensor of a numeric type or string. The output will be of the same tensor type."
```

Select elements of the input tensor based on the indices passed.<br>
The indices are applied to the last axes of the tensor.
