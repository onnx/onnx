```toml
name = "StringConcat"
domain = "ai.onnx"
shape_inference = "defs::text::utils::StringConcatShapeInference_opset20"

[[inputs]]
name = "X"
type = "T"
desc = "Tensor to prepend in concatenation"
differentiation = "non_differentiable"

[[inputs]]
name = "Y"
type = "T"
desc = "Tensor to append in concatenation"
differentiation = "non_differentiable"

[[outputs]]
name = "Z"
type = "T"
desc = "Concatenated string tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = ["tensor(string)"]
desc = "Inputs and outputs must be UTF-8 strings"
```

StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support)
