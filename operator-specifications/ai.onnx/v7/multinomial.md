```toml
name = "Multinomial"
domain = "ai.onnx"
node_determinism = "non_deterministic"

[attributes.dtype]
type = "INT"
desc = "(Optional) The data type for the elements of the output tensor, if not specified, we will use int32."
default = 6

[attributes.sample_size]
type = "INT"
desc = "Number of times to sample."
default = 1

[attributes.seed]
type = "FLOAT"
desc = "(Optional) Seed to the random generator, if not specified we will auto generate one."

[[inputs]]
name = "input"
type = "T1"
desc = "Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch."

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch."

[type_constraints.T1]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain output types to integral tensors."
```

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
