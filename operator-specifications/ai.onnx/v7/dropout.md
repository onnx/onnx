```toml
name = "Dropout"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.ratio]
type = "FLOAT"
desc = "The ratio of random dropout"
default = 0.5

[[outputs]]
name = "output"
type = "T"
desc = "The output."

[[outputs]]
name = "mask"
type = "T"
desc = "The output mask."
option = "optional"
```

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
