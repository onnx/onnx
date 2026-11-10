```toml
name = "FeatureVectorizer"
domain = "ai.onnx.ml"

[attributes.inputdimensions]
type = "INTS"
desc = "The size of each input in the input list"

[[inputs]]
name = "X"
type = "T1"
desc = "An ordered collection of tensors, all with the same element type."
option = "variadic"

[[outputs]]
name = "Y"
type = "tensor(float)"
desc = "The output array, elements ordered as the inputs."

[type_constraints.T1]
types = ["tensor(int32)", "tensor(int64)", "tensor(float)", "tensor(double)"]
desc = "The input type must be a tensor of a numeric type."
```

Concatenates input tensors into one continuous output.<br>
All input shapes are 2-D and are concatenated along the second dimension. 1-D tensors are treated as [1,C].
Inputs are copied to the output maintaining the order of the input arguments.<br>
All inputs must be integers or floats, while the output will be all floating point values.
