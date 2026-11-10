```toml
name = "StringSplit"
domain = "ai.onnx"

[attributes.delimiter]
type = "STRING"
desc = "Delimiter to split on. If left unset or set to the empty string (\"\"), the input is split on consecutive whitespace."

[attributes.maxsplit]
type = "INT"
desc = "Maximum number of splits (from left to right). If left unset (or if the number of possible splits are less than maxsplit), it will make as many splits as possible. Note that the maximum possible number of substrings returned with `maxsplit` specified is `maxsplit+1` since the remaining suffix after the `maxsplit`th split is included in the output."

[[inputs]]
name = "X"
type = "T1"
desc = "Tensor of strings to split."
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T2"
desc = "Tensor of substrings representing the outcome of splitting the strings in the input on the delimiter. Note that to ensure the same number of elements are present in the final rank, this tensor will pad any necessary empty strings."
differentiation = "non_differentiable"

[[outputs]]
name = "Z"
type = "T3"
desc = "The number of substrings generated for each input element."
differentiation = "non_differentiable"

[type_constraints.T1]
types = ["tensor(string)"]
desc = "The input must be a UTF-8 string tensor"

[type_constraints.T2]
types = ["tensor(string)"]
desc = "Tensor of substrings."

[type_constraints.T3]
types = ["tensor(int64)"]
desc = "The number of substrings generated."
```

StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute.

The first output of this operator is a tensor of strings representing the substrings from splitting each input string on the `delimiter` substring. This tensor has one additional rank compared to the input tensor in order to store the substrings for each input element (where the input tensor is not empty). Note that, in order to ensure the same number of elements are present in the final dimension, this tensor will pad empty strings as illustrated in the examples below. Consecutive delimiters are not grouped together and are deemed to delimit empty strings, except if the `delimiter` is unspecified or is the empty string (""). In the case where the `delimiter` is unspecified or the empty string, consecutive whitespace characters are regarded as a single separator and leading or trailing whitespace is removed in the output.

The second output tensor represents the number of substrings generated. `maxsplit` can be used to limit the number of splits performed - after the `maxsplit`th split if the string is not fully split, the trailing suffix of input string after the final split point is also added. For elements where fewer splits are possible than specified in `maxsplit`, it has no effect.
