```toml
name = "ConstantOfShape"
domain = "ai.onnx"

[attributes.value]
type = "TENSOR"
desc = "(Optional) The value of the output elements.Should be a one-element tensor. If not specified, it defaults to a tensor of value 0 and datatype float32"

[[inputs]]
name = "input"
type = "T1"
desc = "1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar. All values must be >= 0."

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor of shape specified by 'input'.If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype defaults to float32."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain input types."

[type_constraints.T2]
types = [
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(bool)",
]
desc = "Constrain output types to be numerics."
```

Generate a tensor with given value and shape.
