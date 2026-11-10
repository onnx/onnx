```toml
name = "BatchNormalization"
domain = "ai.onnx"

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[attributes.momentum]
type = "FLOAT"
desc = "Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum)."
default = 0.9

[attributes.spatial]
type = "INT"
desc = "If true, compute the mean and variance across per activation. If false, compute the mean and variance across per feature over each mini-batch."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size."

[[inputs]]
name = "scale"
type = "T"
desc = "If spatial is true, the dimension of scale is (C). If spatial is false, the dimensions of scale are (C x D1 x ... x Dn)"

[[inputs]]
name = "B"
type = "T"
desc = "If spatial is true, the dimension of bias is (C). If spatial is false, the dimensions of bias are (C x D1 x ... x Dn)"

[[inputs]]
name = "mean"
type = "T"
desc = "If spatial is true, the dimension of the running mean (training) or the estimated mean (testing) is (C). If spatial is false, the dimensions of the running mean (training) or the estimated mean (testing) are (C x D1 x ... x Dn)."

[[inputs]]
name = "var"
type = "T"
desc = "If spatial is true, the dimension of the running variance(training) or the estimated variance (testing) is (C). If spatial is false, the dimensions of the running variance(training) or the estimated variance (testing) are (C x D1 x ... x Dn)."

[[outputs]]
name = "Y"
type = "T"
desc = "The output tensor of the same shape as X"

[[outputs]]
name = "mean"
type = "T"
desc = "The running mean after the BatchNormalization operator."
option = "optional"

[[outputs]]
name = "var"
type = "T"
desc = "The running variance after the BatchNormalization operator."
option = "optional"

[[outputs]]
name = "saved_mean"
type = "T"
desc = "Saved mean used during training to speed up gradient computation."
option = "optional"

[[outputs]]
name = "saved_var"
type = "T"
desc = "Saved variance used during training to speed up gradient computation."
option = "optional"
```

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
