```toml
name = "Normalizer"
domain = "ai.onnx.ml"
shape_inference = false

[attributes.norm]
type = "STRING"
desc = "One of 'MAX,' 'L1,' 'L2'"
default = "MAX"

[[inputs]]
name = "X"
type = "T"
desc = "Data to be encoded, a tensor of shape [N,C] or [C]"

[[outputs]]
name = "Y"
type = "tensor(float)"
desc = "Encoded output data"

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input must be a tensor of a numeric type."
```

Normalize the input. There are three normalization modes, which have the corresponding formulas.
    The function 'abs' and operator '^' are element-wise, while 'max' and 'sum' reduce along the normalization axis:<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(abs(X))<br>
    L2:  Y = X / sqrt(sum(X^2))<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.
