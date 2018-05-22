// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

// Input: box, delta Output: box
ONNX_OPERATOR_SCHEMA(BBoxTransform)
    .NumInputs({3})
    .NumOutputs({1, 2})
    .SetDoc(R"DOC(
Transform proposal bounding boxes to target bounding box using bounding box
    regression deltas.
)DOC")
    .Attr(
        "weights",
        "vector<float> weights [wx, wy, ww, wh] for the deltas",
        AttributeProto::FLOATS,
        std::vector<float>({1.0f, 1.0f, 1.0f, 1.0f}))
    .Attr(
        "apply_scale",
        "bool (default true), transform the boxes to the scaled image space"
        " after applying the bbox deltas."
        "Set to false to match the detectron code, set to true for keypoint"
        " models and for backward compatibility",
        AttributeProto::INT,
        static_cast<int64_t>(1))
    .Attr(
        "correct_transform_coords",
        "bool (default false), Correct bounding box transform coordates,"
        " see bbox_transform() in boxes.py "
        "Set to true to match the detectron code, set to false for backward"
        " compatibility",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Input(
        0,
        "rois",
        "Bounding box proposals in pixel coordinates, "
        "Size (M, 4), format [x1, y1, x2, y2], or"
        "Size (M, 5), format [batch_index, x1, y1, x2, y2]. "
        "If proposals from multiple images in a batch are present, they "
        "should be grouped sequentially and in incremental order.",
        "TINT")
    .Input(
        1,
        "deltas",
        "bounding box translations and scales,"
        "size (M, 4*K), format [dx, dy, dw, dh], K = # classes",
        "TFLOAT")
    .Input(
        2,
        "im_info",
        "Image dimensions, size (batch_size, 3), "
        "format [img_height, img_width, img_scale]",
        "TFLOAT")
    .Output(
        0,
        "box_out",
        "Pixel coordinates of the transformed bounding boxes,"
        "Size (M, 4*K), format [x1, y1, x2, y2]",
        "TINT")
    .Output(
        1,
        "roi_batch_splits",
        "Tensor of shape (batch_size) with each element denoting the number "
        "of RoIs belonging to the corresponding image in batch",
        "TINT")
    .TypeConstraint(
        "TFLOAT",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain score types to float tensors.")
    .TypeConstraint(
        "TINT",
        {"tensor(int32)", "tensor(int64)"},
        "Constrain input and output to float tensors.");


ONNX_OPERATOR_SCHEMA(BoxWithNMSLimit)
    .NumInputs({2, 3})
    .NumOutputs({3, 4, 5, 6})
    .SetDoc(R"DOC(
Apply NMS to each class (except background) and limit the number of
returned boxes.
)DOC")
    .Attr(
        "score_thresh",
        "(float) TEST.SCORE_THRESH",
        AttributeProto::FLOAT,
        0.05f)
    .Attr(
        "nms",
        "(float) TEST.NMS",
        AttributeProto::FLOAT,
        0.3f)
    .Attr(
        "detections_per_im",
        "(int) TEST.DEECTIONS_PER_IM",
        AttributeProto::INT,
        static_cast<int64_t>(100))
    .Attr(
        "soft_nms_enabled",
        "(bool) TEST.SOFT_NMS.ENABLED",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Attr(
        "soft_nms_method",
        "(string) TEST.SOFT_NMS.METHOD",
        AttributeProto::STRING,
        std::string("linear"))
    .Attr(
        "soft_nms_sigma",
        "(float) TEST.SOFT_NMS.SIGMA",
        AttributeProto::FLOAT,
        0.5f)
    .Attr(
        "soft_nms_min_score_thres",
        "(float) Lower bound on updated scores to discard boxes",
        AttributeProto::FLOAT,
        0.001f)
    .Input(
        0,
        "scores",
        "Scores, size (count, num_classes)",
        "TFLOAT")
    .Input(
        1,
        "boxes",
        "Bounding box for each class, size (count, num_classes * 4)",
        "TINT")
    .Input(
        2,
        "batch_splits",
        "Tensor of shape (batch_size) with each element denoting the number "
        "of RoIs/boxes belonging to the corresponding image in batch. "
        "Sum should add up to total count of scores/boxes.",
        "TINT")
    .Output(
        0,
        "scores",
        "Filtered scores, size (n)",
        "TFLOAT")
    .Output(
        1,
        "boxes",
        "Filtered boxes, size (n, 4)",
        "TINT")
    .Output(
        2,
        "classes",
        "Class id for each filtered score/box, size (n)",
        "TINT")
    .Output(
        3,
        "batch_splits",
        "Output batch splits for scores/boxes after applying NMS",
        "TINT")
    .Output(
        4,
        "keeps",
        "Optional filtered indices, size (n)",
        "TINT")
    .Output(
        5,
        "keeps_size",
        "Optional number of filtered indices per class, size (num_classes)",
        "TINT")
    .TypeConstraint(
        "TFLOAT",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain score types to float tensors.")
    .TypeConstraint(
        "TINT",
        {"tensor(int32)", "tensor(int64)"},
        "Constrain input and output to float tensors.");
} // namespace ONNX_NAMESPACE
