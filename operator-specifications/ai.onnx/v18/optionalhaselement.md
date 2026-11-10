```toml
name = "OptionalHasElement"
domain = "ai.onnx"
shape_inference = "defs::optional::utils::OptionalHasElementInference"

[[inputs]]
name = "input"
type = "O"
desc = "The optional input."
option = "optional"

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
desc = "Constrain input type to optional, tensor and sequence types."

[type_constraints.B]
types = ["tensor(bool)"]
desc = "Constrain output to a boolean tensor."
```

Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.
