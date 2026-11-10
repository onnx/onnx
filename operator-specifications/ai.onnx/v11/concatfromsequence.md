```toml
name = "ConcatFromSequence"
domain = "ai.onnx"
shape_inference = "defs::sequence::utils::ConcatFromSequenceShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "Which axis to concat on. Accepted range in `[-r, r - 1]`, where `r` is the rank of input tensors. When `new_axis` is 1, accepted range is `[-r - 1, r]`. "
required = true

[attributes.new_axis]
type = "INT"
desc = "Insert and concatenate on a new axis or not, default 0 means do not insert new axis."
default = 0

[[inputs]]
name = "input_sequence"
type = "S"
desc = "Sequence of tensors for concatenation"

[[outputs]]
name = "concat_result"
type = "T"
desc = "Concatenated tensor"

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
desc = "Constrain input types to any tensor type."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain output types to any tensor type."
```

Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
