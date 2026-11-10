```toml
name = "Transpose"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::TransposeShapeInference_opset25"

[attributes.perm]
type = "INTS"
desc = "A list of integers. By default, reverse the dimensions; otherwise permute the axes according to the values given. Its length must be equal to the rank of the input, and each value must be in the range `[0, rank-1]`."

[type_constraints.T]
types = "all_tensor_types_ir13"
desc = "Constrain input and output types to all tensor types."
```
