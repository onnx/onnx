```toml
name = "NonMaxSuppression"
domain = "ai.onnx"
shape_inference = "defs::object_detection::utils::NonMaxSuppressionShapeInference_opset11"

[[inputs]]
name = "boxes"
type = "tensor(float)"
desc = "An input tensor with shape [num_batches, spatial_dimension, 4]. The single box data format is indicated by center_point_box."

[[inputs]]
name = "scores"
type = "tensor(float)"
desc = "An input tensor with shape [num_batches, num_classes, spatial_dimension]"

[[inputs]]
name = "max_output_boxes_per_class"
type = "tensor(int64)"
desc = "Integer representing the maximum number of boxes to be selected per batch per class. It is a scalar. Default to 0, which means no output."
option = "optional"

[[inputs]]
name = "iou_threshold"
type = "tensor(float)"
desc = "Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Boxes with IoU strictly greater than this threshold are suppressed. It is scalar. Value range [0, 1]. Default to 0."
option = "optional"

[[inputs]]
name = "score_threshold"
type = "tensor(float)"
desc = "Float representing the threshold for deciding when to remove boxes based on score. It is a scalar."
option = "optional"
```
