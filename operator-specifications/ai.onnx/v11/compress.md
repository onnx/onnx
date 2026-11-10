```toml
name = "Compress"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::CompressShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "(Optional) Axis along which to take slices. If not specified, input is flattened before elements being selected. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."

[[inputs]]
name = "input"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[[inputs]]
name = "condition"
type = "T1"
desc = "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. Its length can be less than the input length along the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1."
differentiation = "differentiable"
```
