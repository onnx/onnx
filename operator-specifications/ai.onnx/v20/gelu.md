```toml
name = "Gelu"
domain = "ai.onnx"
node_determinism = "deterministic"
shape_inference = "propagateShapeAndTypeFromFirstInput"
function_body_builders = { 20 = "defs::math::utils::BuildContextDependentFunctionBodyGelu" }

[attributes.approximate]
type = "STRING"
desc = "Gelu approximation algorithm: `\"tanh\"`, `\"none\"`(default).`\"none\"`: do not use approximation.`\"tanh\"`: use tanh approximation."
default = "none"

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```

Gelu takes one input data (Tensor<T>) and produces one
output data (Tensor<T>) where the gaussian error linear units function,
$y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
If the attribute "approximate" is set to "tanh", the function estimation,
$y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
to the tensor elementwise.
