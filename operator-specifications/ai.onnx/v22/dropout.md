```toml
name = "Dropout"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "The input data as Tensor."
differentiation = "differentiable"

[[inputs]]
name = "ratio"
type = "T1"
desc = "The ratio of random dropout, with value in [0, 1). If set to 0, the output would be a simple copy of the input. If it's non-zero, output will be a random dropout of the scaled input, which is typically the case during training. It is an optional value, if not specified it will default to 0.5."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "training_mode"
type = "T2"
desc = "If set to true then it indicates dropout is being used for training. It is an optional value hence unless specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where nothing will be dropped from the input data and if mask is requested as output it will contain all ones."
option = "optional"
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_float_types_ir9"
desc = "Constrain input and output types to float tensors."

[type_constraints.T1]
types = "all_float_types_ir9"
desc = "Constrain input 'ratio' types to float tensors."

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Constrain output 'mask' types to boolean tensors."
```
