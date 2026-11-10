```toml
name = "Upsample"
domain = "ai.onnx"
support_level = "experimental"
shape_inference = false

[attributes.height_scale]
type = "FLOAT"
desc = "The scale along height dimension. It takes value greater than or equal to 1."
required = true

[attributes.mode]
type = "STRING"
desc = "Two interpolation modes: nearest(default), bilinear"
default = "nearest"

[attributes.width_scale]
type = "FLOAT"
desc = "The scale along width dimension. It takes value greater than or equal to 1."
required = true

[[inputs]]
name = "X"
type = "T"
desc = "4-D tensor, [N,C,H,W]"

[[outputs]]
name = "Y"
type = "T"
desc = "4-D tensor after resizing, [N,C,H,W]"

[type_constraints.T]
types = [
    "tensor(bool)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
]
desc = "Constrain output types to bool, int32, int64, float16, float, double tensors."
```

Upsample the input tensor.
The width and height of the output tensor are:
  output_width = floor(input_width * width_scale),
  output_height = floor(input_height * height_scale).
Example:
  Given `data` tensor, width_scale, height_scale, mode,
  Upsample the input 4-D tensor in nearest mode:
  data = [[[
      [1, 2],
      [3, 4]
  ]]]
  width_scale = 2
  height_scale = 2
  mode = "nearest"
  output = [[[
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]
  ]]]
