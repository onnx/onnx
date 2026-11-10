```toml
name = "BatchNormalization"
domain = "ai.onnx"

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero, default is 1e-5f."
default = 1e-05

[attributes.is_test]
type = "INT"
desc = "If set to nonzero, run spatial batch normalization in test mode, default is 0."
default = 0

[attributes.momentum]
type = "FLOAT"
desc = "Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum), default is 0.9f."
default = 0.9

[attributes.spatial]
type = "INT"
desc = "If true, compute the mean and variance across all spatial elements If false, compute the mean and variance across per feature.Default is 1."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size."

[[inputs]]
name = "scale"
type = "T"
desc = "The scale as a 1-dimensional tensor of size C to be applied to the output."

[[inputs]]
name = "B"
type = "T"
desc = "The bias as a 1-dimensional tensor of size C to be applied to the output."

[[inputs]]
name = "mean"
type = "T"
desc = "The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size C."

[[inputs]]
name = "var"
type = "T"
desc = "The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size C."

[[outputs]]
name = "Y"
type = "T"
desc = "The output tensor of the same shape as X."

[[outputs]]
name = "mean"
type = "T"
desc = "The running mean after the BatchNormalization operator. Must be in-place with the input mean. Should not be used for testing."
option = "optional"

[[outputs]]
name = "var"
type = "T"
desc = "The running variance after the BatchNormalization operator. Must be in-place with the input var. Should not be used for testing."
option = "optional"

[[outputs]]
name = "saved_mean"
type = "T"
desc = "Saved mean used during training to speed up gradient computation. Should not be used for testing."
option = "optional"

[[outputs]]
name = "saved_var"
type = "T"
desc = "Saved variance used during training to speed up gradient computation. Should not be used for testing."
option = "optional"
```
