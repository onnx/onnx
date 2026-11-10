```toml
name = "If"
domain = "ai.onnx"
shape_inference = "defs::controlflow::utils::IfInferenceFunction_opset1"

[attributes.else_branch]
type = "GRAPH"
desc = "Graph to run if condition is false. Has N outputs: values you wish to be live-out to the enclosing scope. The number of outputs must match the number of outputs in the then_branch."
required = true

[attributes.then_branch]
type = "GRAPH"
desc = "Graph to run if condition is true. Has N outputs: values you wish to be live-out to the enclosing scope. The number of outputs must match the number of outputs in the else_branch."
required = true

[[inputs]]
name = "cond"
type = "B"
desc = "Condition for the if. The tensor must contain a single element."

[[outputs]]
name = "outputs"
type = "V"
desc = "Values that are live-out to the enclosing scope. The return values in the `then_branch` and `else_branch` must be of the same shape and same data type."
option = "variadic"
homogeneous = false

[type_constraints.V]
types = "all_tensor_types"
desc = "All Tensor types"

[type_constraints.B]
types = ["tensor(bool)"]
desc = "Only bool"
```

If conditional
