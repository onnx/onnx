```toml
name = "RNN"
domain = "ai.onnx"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[type_constraints.T1]
types = ["tensor(int32)"]
desc = "Constrain seq_lens to integer tensor."
```
