```toml
name = "LabelEncoder"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::LabelEncoderShapeInference_opset4"

[attributes.default_float]
type = "FLOAT"
desc = "A float."
default = -0.0

[attributes.default_int64]
type = "INT"
desc = "An integer."
default = -1

[attributes.default_string]
type = "STRING"
desc = "A string."
default = "_Unused"

[attributes.default_tensor]
type = "TENSOR"
desc = "A default tensor. {\"_Unused\"} if values_* has string type, {-1} if values_* has integral type, and {-0.f} if values_* has float type."

[attributes.keys_floats]
type = "FLOATS"
desc = "A list of floats."

[attributes.keys_int64s]
type = "INTS"
desc = "A list of ints."

[attributes.keys_strings]
type = "STRINGS"
desc = "A list of strings."

[attributes.keys_tensor]
type = "TENSOR"
desc = "Keys encoded as a 1D tensor. One and only one of 'keys_*'s should be set."

[attributes.values_floats]
type = "FLOATS"
desc = "A list of floats."

[attributes.values_int64s]
type = "INTS"
desc = "A list of ints."

[attributes.values_strings]
type = "STRINGS"
desc = "A list of strings."

[attributes.values_tensor]
type = "TENSOR"
desc = "Values encoded as a 1D tensor. One and only one of 'values_*'s should be set."

[[inputs]]
name = "X"
type = "T1"
desc = "Input data. It must have the same element type as the keys_* attribute set."

[[outputs]]
name = "Y"
type = "T2"
desc = "Output data. This tensor's element type is based on the values_* attribute set."

[type_constraints.T1]
types = [
    "tensor(string)",
    "tensor(int64)",
    "tensor(float)",
    "tensor(int32)",
    "tensor(int16)",
    "tensor(double)",
]
desc = "The input type is a tensor of any shape."

[type_constraints.T2]
types = [
    "tensor(string)",
    "tensor(int64)",
    "tensor(float)",
    "tensor(int32)",
    "tensor(int16)",
    "tensor(double)",
]
desc = "Output type is determined by the specified 'values_*' attribute."
```

Maps each element in the input tensor to another value.<br>
The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. Note that the 'keys_*' and 'values_*' attributes
must have the same length. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value. The type of the 'default_*'
attribute must match the 'values_*' attribute chosen. <br>
Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
"Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
Float keys with value 'NaN' match any input 'NaN' value regardless of bit
value. If a key is repeated, the last key takes precedence.
