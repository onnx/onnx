```toml
name = "ReverseSequence"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ReverseSequenceShapeInference_opset10"

[attributes.batch_axis]
type = "INT"
desc = "(Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0."
default = 1

[attributes.time_axis]
type = "INT"
desc = "(Optional) Specify which axis is time axis. Must be one of 0 (default), or 1."
default = 0

[[inputs]]
name = "input"
type = "T"
desc = "Tensor of rank r >= 2."

[[inputs]]
name = "sequence_lens"
type = "tensor(int64)"
desc = "Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`."

[[outputs]]
name = "Y"
type = "T"
desc = "Tensor with same shape of input."

[type_constraints.T]
types = "all_tensor_types"
desc = "Input and output types can be of any tensor type."
```

Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
