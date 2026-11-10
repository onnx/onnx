```toml
name = "NegativeLogLikelihoodLoss"
domain = "ai.onnx"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input, weight, and output types to floating-point tensors."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain target to integer types"
```
