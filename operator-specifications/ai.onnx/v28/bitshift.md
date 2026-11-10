```toml
name = "BitShift"
domain = "ai.onnx"
shape_inference = "defs::logical::utils::BitShiftShapeInference_opset28"

[type_constraints.T]
types = [
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
]
desc = "Constrain input and output types to integer tensors."
```

Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side. If the attribute "direction" is "LEFT", bits of binary representation
move toward the left side. The input X is the tensor to be shifted and another
input Y specifies the amounts of shifting. For example, if "direction" is
"RIGHT", X is [1, 4], and Y is [1, 1], the corresponding output Z would be
[0, 2]. If "direction" is "LEFT" with X=[1, 2] and Y=[1, 2], the corresponding
output Z would be [2, 8].

For a signed T the right shift is an arithmetic shift (sign-extending). The
vacated high bits are filled with copies of the sign bit, so a negative X stays
negative. For a signed T a left shift can move bits into and past the sign bit,
and bits shifted past the sign bit are discarded.

If Y is negative, or is greater than or equal to the number of bits of T, then
the result is whatever the sign bit extension alone produces: -1 for a right
shift on a negative X, where the fill is a sign bit of 1, and 0 in every other
case.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
