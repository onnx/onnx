```toml
name = "Celu"
domain = "ai.onnx"
node_determinism = "deterministic"

[attributes.alpha]
type = "FLOAT"
desc = "The Alpha value in Celu formula which control the shape of the unit. The default value is 1.0."
default = 1.0

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = ["tensor(float)"]
desc = "Constrain input and output types to float32 tensors."
```

Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
