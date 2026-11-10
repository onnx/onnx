```toml
name = "IsNaN"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::IsNaNShapeInference_opset9"

[[inputs]]
name = "X"
type = "T1"
desc = "input"

[[outputs]]
name = "Y"
type = "T2"
desc = "output"

[type_constraints.T1]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Constrain output types to boolean tensors."
```

Returns which elements of the input are NaN.
