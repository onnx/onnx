```toml
name = "EyeLike"
domain = "ai.onnx"
shape_inference = "defs::generator::utils::EyeLikeShapeInference_opset22"

[attributes.dtype]
type = "INT"
desc = "(Optional) The data type for the elements of the output tensor. If not specified, the data type of the input tensor T1 is used."

[attributes.k]
type = "INT"
desc = "(Optional) Index of the diagonal to be populated with ones. Default is 0. If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, k > 0 populates an upper diagonal,  and k < 0 populates a lower diagonal."
default = 0

[type_constraints.T1]
types = "all_non_complex_numeric_types_plus_bool_ir4"
desc = "Constrain input types. Strings and complex are not supported."

[type_constraints.T2]
types = "all_non_complex_numeric_types_plus_bool_ir4"
desc = "Constrain output types. Strings and complex are not supported."
```
