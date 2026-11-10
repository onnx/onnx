```toml
name = "BitShift"
domain = "ai.onnx"

[attributes.direction]
type = "STRING"
desc = "Direction of moving bits. It can be either \"RIGHT\" (for right shift) or \"LEFT\" (for left shift)."
required = true

[[inputs]]
name = "X"
type = "T"
desc = "First operand, input to be shifted."
differentiation = "non_differentiable"

[[inputs]]
name = "Y"
type = "T"
desc = "Second operand, amounts of shift."
differentiation = "non_differentiable"

[[outputs]]
name = "Z"
type = "T"
desc = "Output tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = ["tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)"]
desc = "Constrain input and output types to integer tensors."
```

Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
