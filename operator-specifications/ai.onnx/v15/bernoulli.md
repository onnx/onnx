```toml
name = "Bernoulli"
domain = "ai.onnx"
node_determinism = "non_deterministic"

[attributes.dtype]
type = "INT"
desc = "The data type for the elements of the output tensor. if not specified, we will use the data type of the input tensor."

[attributes.seed]
type = "FLOAT"
desc = "(Optional) Seed to the random generator, if not specified we will auto generate one."

[[inputs]]
name = "input"
type = "T1"
desc = "All values in input have to be in the range:[0, 1]."

[[outputs]]
name = "output"
type = "T2"
desc = "The returned output tensor only has values 0 or 1, same shape as input tensor."

[type_constraints.T1]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = "all_non_complex_numeric_types_plus_bool_ir4"
desc = "Constrain output types to all numeric tensors and bool tensors."
```

Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
