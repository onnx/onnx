```toml
name = "Pad"
domain = "ai.onnx"

[attributes.mode]
type = "STRING"
desc = "Three modes: constant(default), reflect, edge"
default = "constant"

[attributes.paddings]
type = "INTS"
desc = "List of integers indicate the padding element count at the beginning and end of each axis, for 2D it is the number of pixel. `paddings` rank should be double of the input's rank. `paddings` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
required = true

[attributes.value]
type = "FLOAT"
desc = "One float, indicates the value to be filled, default is 0"
default = 0.0

[[inputs]]
name = "data"
type = "T"
desc = "Input tensor."

[[outputs]]
name = "output"
type = "T"
desc = "Tensor after padding."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Given `data` tensor, paddings, mode, and value.
Example:
  Insert 0 paddings to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  paddings = [0, 0, 2, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
