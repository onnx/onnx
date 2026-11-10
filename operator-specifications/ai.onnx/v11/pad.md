```toml
name = "Pad"
domain = "ai.onnx"

[attributes.mode]
type = "STRING"
desc = "Supported modes: `constant`(default), `reflect`, `edge`"
default = "constant"

[[inputs]]
name = "data"
type = "T"
desc = "Input tensor."

[[inputs]]
name = "pads"
type = "tensor(int64)"
desc = "Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. `pads` should be a 1D tensor of shape [2 * input_rank]. `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end, the number of pad values added at the end of axis `i`."

[[inputs]]
name = "constant_value"
type = "T"
desc = "(Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0)."
option = "optional"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output to only numeric types."
```

Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output =
  [
      [0.0, 0.0, 1.0, 1.2],
      [0.0, 0.0, 2.3, 3.4],
      [0.0, 0.0, 4.5, 5.7],
  ]


Example 2 (`reflect` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output =
  [
      [1.0, 1.2, 1.0, 1.2],
      [2.3, 3.4, 2.3, 3.4],
      [4.5, 5.7, 4.5, 5.7],
  ]


Example 3 (`edge` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output =
  [
      [1.0, 1.0, 1.0, 1.2],
      [2.3, 2.3, 2.3, 3.4],
      [4.5, 4.5, 4.5, 5.7],
  ]
