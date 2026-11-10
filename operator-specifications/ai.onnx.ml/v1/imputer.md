```toml
name = "Imputer"
domain = "ai.onnx.ml"
shape_inference = false

[attributes.imputed_value_floats]
type = "FLOATS"
desc = "Value(s) to change to"

[attributes.imputed_value_int64s]
type = "INTS"
desc = "Value(s) to change to."

[attributes.replaced_value_float]
type = "FLOAT"
desc = "A value that needs replacing."
default = 0.0

[attributes.replaced_value_int64]
type = "INT"
desc = "A value that needs replacing."
default = 0

[[inputs]]
name = "X"
type = "T"
desc = "Data to be processed."

[[outputs]]
name = "Y"
type = "T"
desc = "Imputed output data"

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input type must be a tensor of a numeric type, either [N,C] or [C]. The output type will be of the same tensor type and shape."
```

Replaces inputs that equal one value with another, leaving all other elements alone.<br>
This operator is typically used to replace missing values in situations where they have a canonical
representation, such as -1, 0, NaN, or some extreme value.<br>
One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
which one depends on whether floats or integers are being processed.<br>
The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
