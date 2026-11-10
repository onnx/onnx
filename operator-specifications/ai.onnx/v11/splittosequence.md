```toml
name = "SplitToSequence"
domain = "ai.onnx"
shape_inference = "defs::sequence::utils::SplitToSequenceShapeInference"

[attributes.axis]
type = "INT"
desc = "Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1]."
default = 0

[attributes.keepdims]
type = "INT"
desc = "Keep the split dimension or not. Default 1, which means we keep split dimension. If input 'split' is specified, this attribute is ignored."
default = 1

[[inputs]]
name = "input"
type = "T"
desc = "The tensor to split"

[[inputs]]
name = "split"
type = "I"
desc = "Length of each output. It can be either a scalar(tensor of empty shape), or a 1-D tensor. All values must be >= 0. "
option = "optional"

[[outputs]]
name = "output_sequence"
type = "S"
desc = "One or more outputs forming a sequence of tensors after splitting"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input types to all tensor types."

[type_constraints.I]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain split size to integral tensor."

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
desc = "Constrain output types to all tensor types."
```

Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If the argument `split' is not specified, a default scalar value of 1
is used as the value of `split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.
