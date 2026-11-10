```toml
name = "CumSum"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[type_constraints.T]
types = "numeric_types_for_math_reduction_ir4"
desc = "Constrain input and output types to numeric tensors."

[type_constraints.T2]
types = ["tensor(int32)", "tensor(int64)"]
desc = "axis tensor can be int32 or int64 only"
```
