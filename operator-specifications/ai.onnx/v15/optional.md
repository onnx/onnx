```toml
name = "Optional"
domain = "ai.onnx"

[attributes.type]
type = "TYPE_PROTO"
desc = "Type of the element in the optional output"

[[inputs]]
name = "input"
type = "V"
desc = "The input element."
option = "optional"

[[outputs]]
name = "output"
type = "O"
desc = "The optional output enclosing the input element."

[type_constraints.V]
types = [
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(string)",
    "tensor(bool)",
    "tensor(complex64)",
    "tensor(complex128)",
    "seq(tensor(uint8))",
    "seq(tensor(uint16))",
    "seq(tensor(uint32))",
    "seq(tensor(uint64))",
    "seq(tensor(int8))",
    "seq(tensor(int16))",
    "seq(tensor(int32))",
    "seq(tensor(int64))",
    "seq(tensor(float16))",
    "seq(tensor(float))",
    "seq(tensor(double))",
    "seq(tensor(string))",
    "seq(tensor(bool))",
    "seq(tensor(complex64))",
    "seq(tensor(complex128))",
]
desc = "Constrain input type to all tensor and sequence types."

[type_constraints.O]
types = [
    "optional(seq(tensor(uint8)))",
    "optional(seq(tensor(uint16)))",
    "optional(seq(tensor(uint32)))",
    "optional(seq(tensor(uint64)))",
    "optional(seq(tensor(int8)))",
    "optional(seq(tensor(int16)))",
    "optional(seq(tensor(int32)))",
    "optional(seq(tensor(int64)))",
    "optional(seq(tensor(float16)))",
    "optional(seq(tensor(float)))",
    "optional(seq(tensor(double)))",
    "optional(seq(tensor(string)))",
    "optional(seq(tensor(bool)))",
    "optional(seq(tensor(complex64)))",
    "optional(seq(tensor(complex128)))",
    "optional(tensor(uint8))",
    "optional(tensor(uint16))",
    "optional(tensor(uint32))",
    "optional(tensor(uint64))",
    "optional(tensor(int8))",
    "optional(tensor(int16))",
    "optional(tensor(int32))",
    "optional(tensor(int64))",
    "optional(tensor(float16))",
    "optional(tensor(float))",
    "optional(tensor(double))",
    "optional(tensor(string))",
    "optional(tensor(bool))",
    "optional(tensor(complex64))",
    "optional(tensor(complex128))",
]
desc = "Constrain output type to all optional tensor or optional sequence types."
```

Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
