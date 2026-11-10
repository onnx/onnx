```toml
name = "LabelEncoder"
domain = "ai.onnx.ml"

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

[attributes.keys_floats]
type = "FLOATS"
desc = "A list of floats."

[attributes.keys_int64s]
type = "INTS"
desc = "A list of ints."

[attributes.keys_strings]
type = "STRINGS"
desc = "A list of strings. One and only one of 'keys_*'s should be set."

[attributes.values_floats]
type = "FLOATS"
desc = "A list of floats."

[attributes.values_int64s]
type = "INTS"
desc = "A list of ints."

[attributes.values_strings]
type = "STRINGS"
desc = "A list of strings. One and only one of 'value_*'s should be set."

[[inputs]]
name = "X"
type = "T1"
desc = "Input data. It can be either tensor or scalar."

[[outputs]]
name = "Y"
type = "T2"
desc = "Output data."

[type_constraints.T1]
types = ["tensor(string)", "tensor(int64)", "tensor(float)"]
desc = "The input type is a tensor of any shape."

[type_constraints.T2]
types = ["tensor(string)", "tensor(int64)", "tensor(float)"]
desc = "Output type is determined by the specified 'values_*' attribute."
```

Maps each element in the input tensor to another value.<br>
The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value.<br>
Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
"Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
For key look-up, bit-wise comparison is used so even a float NaN can be
mapped to a value in 'values_*' attribute.<br>
