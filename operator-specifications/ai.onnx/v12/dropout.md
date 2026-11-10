```toml
name = "Dropout"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::DropoutShapeInference_opset12"

[attributes.seed]
type = "INT"
desc = "(Optional) Seed to the random generator, if not specified we will auto generate one."

[[inputs]]
name = "data"
type = "T"
desc = "The input data as Tensor."

[[inputs]]
name = "ratio"
type = "T1"
desc = "The ratio of random dropout, with value in [0, 1). If this input was not set, or if it was set to 0, the output would be a simple copy of the input. If it's non-zero, output will be a random dropout of the scaled input, which is typically the case during training. It is an optional value, if not specified it will default to 0.5."
option = "optional"

[[inputs]]
name = "training_mode"
type = "T2"
desc = "If set to true then it indicates dropout is being used for training. It is an optional value hence unless specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where nothing will be dropped from the input data and if mask is requested as output it will contain all ones."
option = "optional"

[[outputs]]
name = "output"
type = "T"
desc = "The output."

[[outputs]]
name = "mask"
type = "T2"
desc = "The output mask."
option = "optional"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."

[type_constraints.T1]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input 'ratio' types to float tensors."

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Constrain output 'mask' types to boolean tensors."
```

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
