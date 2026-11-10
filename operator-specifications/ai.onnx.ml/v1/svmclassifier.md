```toml
name = "SVMClassifier"
domain = "ai.onnx.ml"

[attributes.classlabels_ints]
type = "INTS"
desc = "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined."

[attributes.classlabels_strings]
type = "STRINGS"
desc = "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined."

[attributes.coefficients]
type = "FLOATS"
desc = ""

[attributes.kernel_params]
type = "FLOATS"
desc = "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel."

[attributes.kernel_type]
type = "STRING"
desc = "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'."
default = "LINEAR"

[attributes.post_transform]
type = "STRING"
desc = "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'"
default = "NONE"

[attributes.prob_a]
type = "FLOATS"
desc = "First set of probability coefficients."

[attributes.prob_b]
type = "FLOATS"
desc = "Second set of probability coefficients. This array must be same size as prob_a.<br>If these are provided then output Z are probability estimates, otherwise they are raw scores."

[attributes.rho]
type = "FLOATS"
desc = ""

[attributes.support_vectors]
type = "FLOATS"
desc = ""

[attributes.vectors_per_class]
type = "INTS"
desc = ""

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
desc = "Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores."

[type_constraints.T1]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input must be a tensor of a numeric type, either [C] or [N,C]."

[type_constraints.T2]
types = ["tensor(string)", "tensor(int64)"]
desc = "The output type will be a tensor of strings or integers, depending on which of the classlabels_* attributes is used. Its size will match the batch size of the input."
```

Support Vector Machine classifier
