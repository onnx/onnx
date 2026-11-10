```toml
name = "NonMaxSuppression"
domain = "ai.onnx"
shape_inference = "defs::object_detection::utils::NonMaxSuppressionShapeInference_opset10"

[attributes.center_point_box]
type = "INT"
desc = "Integer indicate the format of the box data. The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners and the coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute. Mostly used for TF models. 1 - the box data is supplied as [x_center, y_center, width, height]. Mostly used for Pytorch models."
default = 0

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
desc = "Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1]. Default to 0."
option = "optional"

[[inputs]]
name = "score_threshold"
type = "tensor(float)"
desc = "Float representing the threshold for deciding when to remove boxes based on score. It is a scalar."
option = "optional"

[[outputs]]
name = "selected_indices"
type = "tensor(int64)"
desc = "selected indices from the boxes tensor. [num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index]."
```

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Boxes are suppressed if their IOU with a previously selected box is strictly greater than iou_threshold (i.e., boxes with IOU exactly equal to the threshold are kept).
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
