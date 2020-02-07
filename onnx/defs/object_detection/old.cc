// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

static const char* NonMaxSuppression_doc = R"DOC(
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    NonMaxSuppression,
    10,
    OpSchema()
        .Input(
            0,
            "boxes",
            "An input tensor with shape [num_batches, spatial_dimension, 4]. The single box data format is indicated by center_point_box.",
            "tensor(float)")
        .Input(
            1,
            "scores",
            "An input tensor with shape [num_batches, num_classes, spatial_dimension]",
            "tensor(float)")
        .Input(
            2,
            "max_output_boxes_per_class",
            "Integer representing the maximum number of boxes to be selected per batch per class. It is a scalar. Default to 0, which means no output.",
            "tensor(int64)",
            OpSchema::Optional)
        .Input(
            3,
            "iou_threshold",
            "Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1]. Default to 0.",
            "tensor(float)",
            OpSchema::Optional)
        .Input(
            4,
            "score_threshold",
            "Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.",
            "tensor(float)",
            OpSchema::Optional)
        .Output(
            0,
            "selected_indices",
            "selected indices from the boxes tensor. [num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index].",
            "tensor(int64)")
        .Attr(
            "center_point_box",
            "Integer indicate the format of the box data. The default is 0. "
            "0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners "
            "and the coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute. Mostly used for TF models. "
            "1 - the box data is supplied as [x_center, y_center, width, height]. Mostly used for Pytorch models.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .SetDoc(NonMaxSuppression_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto selected_indices_type =
              ctx.getOutputType(0)->mutable_tensor_type();
          selected_indices_type->set_elem_type(
              ::ONNX_NAMESPACE::TensorProto_DataType::
                  TensorProto_DataType_INT64);
        }));

} // namespace ONNX_NAMESPACE
