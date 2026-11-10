```toml
name = "Flatten"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::FlattenShapeInference_opset13"

[[inputs]]
name = "input"
type = "T"
desc = "A tensor of rank >= axis."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "A 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output to all tensor types."
```
