```toml
name = "CenterCropPad"
domain = "ai.onnx"
node_determinism = "deterministic"
shape_inference = "defs::tensor::utils::CenterCropPadShapeInference_opset18"
function_body_builders = { 18 = "defs::tensor::utils::CenterCropPadFunctionBody_opset18" }

[attributes.axes]
type = "INTS"
desc = "If provided, it specifies a subset of axes that 'shape' refer to. If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). Negative value means counting dimensions from the back. Accepted range is [-r, r-1], where r = rank(data). Behavior is undefined if an axis is repeated."

[[inputs]]
name = "input_data"
type = "T"
desc = "Input to extract the centered crop from."
differentiation = "differentiable"

[[inputs]]
name = "shape"
type = "Tind"
desc = "1-D tensor representing the cropping window dimensions."
differentiation = "non_differentiable"

[[outputs]]
name = "output_data"
type = "T"
desc = "Output data."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```

Center crop or pad an input to given dimensions.

The crop/pad dimensions can be specified for a subset of the `axes`; unspecified dimensions will remain unchanged.

If the input dimensions are larger than the target crop dimensions, a centered cropping window will be extracted
from the input. The starting value for the cropping window is rounded down, which means that if the difference
between the input shape and the crop shape is odd, the cropping window will be shifted half a pixel to the left
of the input center.

If the input dimensions are smaller than the target crop dimensions, the input will be padded equally on both sides
to center it in the output. In cases where the total number of padding pixels is odd, an additional pixel will be
added to the right side.

The padding value used is zero.
