```toml
name = "RegexFullMatch"
domain = "ai.onnx"

[attributes.pattern]
type = "STRING"
desc = "Regex pattern to match on. This must be valid RE2 syntax."

[[inputs]]
name = "X"
type = "T1"
desc = "Tensor with strings to match on."
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T2"
desc = "Tensor of bools indicating if each input string fully matches the regex pattern specified."
differentiation = "non_differentiable"

[type_constraints.T1]
types = ["tensor(string)"]
desc = "Inputs must be UTF-8 strings"

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Outputs are bools and are True where there is a full regex match and False otherwise."
```

RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.
