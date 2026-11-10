```toml
name = "SVMRegressor"
domain = "ai.onnx.ml"

[attributes.coefficients]
type = "FLOATS"
desc = "Support vector coefficients."

[attributes.kernel_params]
type = "FLOATS"
desc = "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel."

[attributes.kernel_type]
type = "STRING"
desc = "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'."
default = "LINEAR"

[attributes.n_supports]
type = "INT"
desc = "The number of support vectors."
default = 0

[attributes.one_class]
type = "INT"
desc = "Flag indicating whether the regression is a one-class SVM or not."
default = 0

[attributes.post_transform]
type = "STRING"
desc = "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'"
default = "NONE"

[attributes.rho]
type = "FLOATS"
desc = ""

[attributes.support_vectors]
type = "FLOATS"
desc = "Chosen support vectors"

[[inputs]]
name = "X"
type = "T"
desc = "Data to be regressed."

[[outputs]]
name = "Y"
type = "tensor(float)"
desc = "Regression outputs (one score per target per example)."

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input type must be a tensor of a numeric type, either [C] or [N,C]."
```

Support Vector Machine regression prediction and one-class SVM anomaly detection.
