```toml
name = "BatchNormalization"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[attributes.momentum]
type = "FLOAT"
desc = "Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum)."
default = 0.9

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1"
differentiation = "differentiable"

[[inputs]]
name = "scale"
type = "T"
desc = "Scale tensor of shape (C)."
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Bias tensor of shape (C)."
differentiation = "differentiable"

[[inputs]]
name = "mean"
type = "T"
desc = "running (training) or estimated (testing) mean tensor of shape (C)."
differentiation = "differentiable"

[[inputs]]
name = "var"
type = "T"
desc = "running (training) or estimated (testing) variance tensor of shape (C)."
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "The output tensor of the same shape as X"
differentiation = "differentiable"

[[outputs]]
name = "mean"
type = "T"
desc = "The running mean after the BatchNormalization operator."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "var"
type = "T"
desc = "The running variance after the BatchNormalization operator."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "saved_mean"
type = "T"
desc = "Saved mean used during training to speed up gradient computation."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "saved_var"
type = "T"
desc = "Saved variance used during training to speed up gradient computation."
option = "optional"
differentiation = "non_differentiable"
```

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
