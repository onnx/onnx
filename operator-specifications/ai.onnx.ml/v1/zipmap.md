```toml
name = "ZipMap"
domain = "ai.onnx.ml"

[attributes.classlabels_int64s]
type = "INTS"
desc = "The keys when using int keys.<br>One and only one of the 'classlabels_*' attributes must be defined."

[attributes.classlabels_strings]
type = "STRINGS"
desc = "The keys when using string keys.<br>One and only one of the 'classlabels_*' attributes must be defined."

[[inputs]]
name = "X"
type = "tensor(float)"
desc = "The input values"

[[outputs]]
name = "Z"
type = "T"
desc = "The output map"

[type_constraints.T]
types = ["seq(map(string, float))", "seq(map(int64, float))"]
desc = "The output will be a sequence of string or integer maps to float."
```

Creates a map from the input and the attributes.<br>
The values are provided by the input tensor, while the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
