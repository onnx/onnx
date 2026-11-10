```toml
name = "SequenceEmpty"
domain = "ai.onnx"
shape_inference = "defs::sequence::utils::SequenceEmptyShapeInference_opset11"

[attributes.dtype]
type = "INT"
desc = "(Optional) The data type of the tensors in the output sequence. The default type is 'float'."

[[outputs]]
name = "output"
type = "S"
desc = "Empty sequence."

[type_constraints.S]
types = [
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
desc = "Constrain output types to any tensor type."
```

Construct an empty tensor sequence, with given data type.
