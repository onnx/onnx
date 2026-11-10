```toml
name = "MaxRoiPool"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::roiPoolTypeShapeInference_opset1"

[attributes.pooled_shape]
type = "INTS"
desc = "ROI pool output shape (height, width)."
required = true

[attributes.spatial_scale]
type = "FLOAT"
desc = "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling."
default = 1.0

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data."
differentiation = "differentiable"

[[inputs]]
name = "rois"
type = "T"
desc = "RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...]."
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1])."
differentiation = "differentiable"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

ROI max pool consumes an input tensor X and region of interests (RoIs) to
apply max pooling across each RoI, to produce output 4-D tensor of shape
(num_rois, channels, pooled_shape[0], pooled_shape[1]).
