```toml
name = "LabelEncoder"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::LabelEncoderShapeInference_opset1"

[attributes.classes_strings]
type = "STRINGS"
desc = "A list of labels."

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
desc = "Input data."

[[outputs]]
name = "Y"
type = "T2"
desc = "Output data. If strings are input, the output values are integers, and vice versa."

[type_constraints.T1]
types = ["tensor(string)", "tensor(int64)"]
desc = "The input type must be a tensor of integers or strings, of any shape."

[type_constraints.T2]
types = ["tensor(string)", "tensor(int64)"]
desc = "The output type will be a tensor of strings or integers, and will have the same shape as the input."
```

Converts strings to integers and vice versa.<br>
If the string default value is set, it will convert integers to strings.
If the int default value is set, it will convert strings to integers.<br>
Each operator converts either integers to strings or strings to integers, depending
on which default value attribute is provided. Only one default value attribute
should be defined.<br>
When converting from integers to strings, the string is fetched from the
'classes_strings' list, by simple indexing.<br>
When converting from strings to integers, the string is looked up in the list
and the index at which it is found is used as the converted value.
