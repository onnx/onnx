```toml
name = "MaxUnpool"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::maxUnpoolShapeInference"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[type_constraints.T2]
types = ["tensor(int64)"]
desc = "Constrain index tensor to int64"
```
