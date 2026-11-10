```toml
name = "Expand"
domain = "ai.onnx"
shape_inference = "defs::math::utils::ExpandShapeInference_opset8"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"

[[inputs]]
name = "shape"
type = "tensor(int64)"
desc = "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensors."
```

Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
