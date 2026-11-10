```toml
name = "Pow"
domain = "ai.onnx"
shape_inference = "defs::math::utils::PowShapeInference_opset12"

[[inputs]]
name = "X"
type = "T"
desc = "First operand, base of the exponent."

[[inputs]]
name = "Y"
type = "T1"
desc = "Second operand, power of the exponent."

[type_constraints.T]
types = ["tensor(int32)", "tensor(int64)", "tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input X and output types to float/int tensors."

[type_constraints.T1]
types = "all_numeric_types"
desc = "Constrain input Y types to float/int tensors."
```
