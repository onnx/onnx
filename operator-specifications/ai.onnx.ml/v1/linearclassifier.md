```toml
name = "LinearClassifier"
domain = "ai.onnx.ml"

[attributes.classlabels_ints]
type = "INTS"
desc = "Class labels when using integer labels. One and only one 'classlabels' attribute must be defined."

[attributes.classlabels_strings]
type = "STRINGS"
desc = "Class labels when using string labels. One and only one 'classlabels' attribute must be defined."

[attributes.coefficients]
type = "FLOATS"
desc = "A collection of weights of the model(s)."
required = true

[attributes.intercepts]
type = "FLOATS"
desc = "A collection of intercepts."

[attributes.multi_class]
type = "INT"
desc = "Indicates whether to do OvR or multinomial (0=OvR is the default)."
default = 0

[attributes.post_transform]
type = "STRING"
desc = "Indicates the transform to apply to the scores vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'"
default = "NONE"

[[inputs]]
name = "X"
type = "T1"
desc = "Data to be classified."

[[outputs]]
name = "Y"
type = "T2"
desc = "Classification outputs (one class per example)."

[[outputs]]
name = "Z"
type = "tensor(float)"
desc = "Classification scores ([N,E] - one score for each class and example"

[type_constraints.T1]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input must be a tensor of a numeric type, and of shape [N,C] or [C]. In the latter case, it will be treated as [1,C]"

[type_constraints.T2]
types = ["tensor(string)", "tensor(int64)"]
desc = "The output will be a tensor of strings or integers."
```

Linear classifier
