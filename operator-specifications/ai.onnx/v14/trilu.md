```toml
name = "Trilu"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::TriluShapeInference_opset14"

[attributes.upper]
type = "INT"
desc = "Boolean. Indicates whether upper or lower part of matrix is retained. Default is true."
default = 1

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor of rank 2 or higher."
differentiation = "differentiable"

[[inputs]]
name = "k"
type = "tensor(int64)"
desc = "A 0-D tensor containing a single value corresponding to the number diagonals above or below the main diagonal to exclude or include. Default value is 0 if it's not specified."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor of the same type and shape as the input tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```

Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute "upper" determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the "upper" attribute is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
