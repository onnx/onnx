```toml
name = "Or"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "If set, defines the broadcast dimensions."

[attributes.broadcast]
type = "INT"
desc = "Enable broadcasting"
default = 0

[[inputs]]
name = "A"
type = "T"
desc = "Left input tensor for the logical operator."

[[inputs]]
name = "B"
type = "T"
desc = "Right input tensor for the logical operator."

[[outputs]]
name = "C"
type = "T1"
desc = "Result tensor."

[type_constraints.T]
types = ["tensor(bool)"]
desc = "Constrain input to boolean tensor."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain output to boolean tensor."
```

Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
