```toml
name = "RoiAlign"
domain = "ai.onnx"
shape_inference = "defs::object_detection::utils::RoiAlignShapeInference_opset16"

[attributes.coordinate_transformation_mode]
type = "STRING"
desc = "Allowed values are 'half_pixel' and 'output_half_pixel'. Use the value 'half_pixel' to pixel shift the input coordinates by -0.5 (the recommended behavior). Use the value 'output_half_pixel' to omit the pixel shift for the input (use this for a backward-compatible behavior)."
default = "half_pixel"

[attributes.mode]
type = "STRING"
desc = "The pooling method. Two modes are supported: 'avg' and 'max'. Default is 'avg'."
default = "avg"

[attributes.output_height]
type = "INT"
desc = "default 1; Pooled output Y's height."
default = 1

[attributes.output_width]
type = "INT"
desc = "default 1; Pooled output Y's width."
default = 1

[attributes.sampling_ratio]
type = "INT"
desc = "Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If == 0, then an adaptive number of grid points are used (computed as ceil(roi_width / output_width), and likewise for height). Default is 0."
default = 0

[attributes.spatial_scale]
type = "FLOAT"
desc = "Multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling, i.e., spatial scale of the input feature map X relative to the input image. E.g.; default is 1.0f. "
default = 1.0
```
