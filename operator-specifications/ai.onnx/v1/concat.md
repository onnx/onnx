```toml
name = "Concat"
domain = "ai.onnx"
shape_inference = false

[attributes.axis]
type = "INT"
desc = "Which axis to concat on.  Default value is 1."

[[inputs]]
name = "inputs"
type = "T"
desc = "List of tensors for concatenation"
option = "variadic"

[[outputs]]
name = "concat_result"
type = "T"
desc = "Concatenated tensor"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain output types to float tensors."
```

Concatenate a list of tensors into a single tensor
