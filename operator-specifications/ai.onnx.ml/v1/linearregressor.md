```toml
name = "LinearRegressor"
domain = "ai.onnx.ml"
shape_inference = false

[attributes.coefficients]
type = "FLOATS"
desc = "Weights of the model(s)."

[attributes.intercepts]
type = "FLOATS"
desc = "Weights of the intercepts, if used."

[attributes.post_transform]
type = "STRING"
desc = "Indicates the transform to apply to the regression output vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'"
default = "NONE"

[attributes.targets]
type = "INT"
desc = "The total number of regression targets, 1 if not defined."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "Data to be regressed."

[[outputs]]
name = "Y"
type = "tensor(float)"
desc = "Regression outputs (one per target, per example)."

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input must be a tensor of a numeric type."
```

Generalized linear regression evaluation.<br>
If targets is set to 1 (default) then univariate regression is performed.<br>
If targets is set to M then M sets of coefficients must be passed in as a sequence
and M results will be output for each input n in N.<br>
The coefficients array is of length n, and the coefficients for each target are contiguous.
Intercepts are optional but if provided must match the number of targets.
