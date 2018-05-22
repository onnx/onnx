// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

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
        "T1")
    .Input(
        1,
        "boxes",
        "Bounding box for each class, size (count, num_classes * 4)",
        "T2")
    .Input(
        2,
        "batch_splits",
        "Tensor of shape (batch_size) with each element denoting the number "
        "of RoIs/boxes belonging to the corresponding image in batch. "
        "Sum should add up to total count of scores/boxes.",
        "T2")
    .Output(
        0,
        "scores",
        "Filtered scores, size (n)",
        "T1")
    .Output(
        1,
        "boxes",
        "Filtered boxes, size (n, 4)",
        "T2")
    .Output(
        2,
        "classes",
        "Class id for each filtered score/box, size (n)",
        "T2")
    .Output(
        3,
        "batch_splits",
        "Output batch splits for scores/boxes after applying NMS",
        "T2")
    .Output(
        4,
        "keeps",
        "Optional filtered indices, size (n)",
        "T2")
    .Output(
        5,
        "keeps_size",
        "Optional number of filtered indices per class, size (num_classes)",
        "T2")
    .TypeConstraint(
        "T1",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain score types to float tensors.")
    .TypeConstraint(
        "T2",
        {"tensor(int32)", "tensor(int64)"},
        "Constrain input and output to float tensors.");
} // namespace ONNX_NAMESPACE
