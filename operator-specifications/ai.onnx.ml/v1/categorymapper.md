```toml
name = "CategoryMapper"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::CategoryMapperShapeInference_opset1"

[attributes.cats_int64s]
type = "INTS"
desc = "The integers of the map. This sequence must be the same length as the 'cats_strings' sequence."

[attributes.cats_strings]
type = "STRINGS"
desc = "The strings of the map. This sequence must be the same length as the 'cats_int64s' sequence"

[attributes.default_int64]
type = "INT"
desc = "An integer to use when an input string value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined."
default = -1

[attributes.default_string]
type = "STRING"
desc = "A string to use when an input integer value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined."
default = "_Unused"

[[inputs]]
name = "X"
type = "T1"
desc = "Input data"

[[outputs]]
name = "Y"
type = "T2"
desc = "Output data. If strings are input, the output values are integers, and vice versa."

[type_constraints.T1]
types = ["tensor(string)", "tensor(int64)"]
desc = "The input must be a tensor of strings or integers, either [N,C] or [C]."

[type_constraints.T2]
types = ["tensor(string)", "tensor(int64)"]
desc = "The output is a tensor of strings or integers. Its shape will be the same as the input shape."
```

Converts strings to integers and vice versa.<br>
Two sequences of equal length are used to map between integers and strings,
with strings and integers at the same index detailing the mapping.<br>
Each operator converts either integers to strings or strings to integers, depending
on which default value attribute is provided. Only one default value attribute
should be defined.<br>
If the string default value is set, it will convert integers to strings.
If the int default value is set, it will convert strings to integers.
