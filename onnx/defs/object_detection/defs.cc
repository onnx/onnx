// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

static const char* RoiAlign_ver1_doc = R"DOC(
Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RoiAlign,
    10,
    OpSchema()
      .SetDoc(RoiAlign_ver1_doc)
      .Attr(
          "spatial_scale",
          "Multiplicative spatial scale factor to translate ROI coordinates "
          "from their input spatial scale to the scale used when pooling, "
          "i.e., spatial scale of the input feature map X relative to the "
          "input image. E.g.; default is 1.0f. ",
          AttributeProto::FLOAT,
          1.f)
      .Attr(
          "output_height",
          "default 1; Pooled output Y's height.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "output_width",
          "default 1; Pooled output Y's width.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "sampling_ratio",
          "Number of sampling points in the interpolation grid used to compute "
          "the output value of each pooled output bin. If > 0, then exactly "
          "sampling_ratio x sampling_ratio grid points are used. If == 0, then "
          "an adaptive number of grid points are used (computed as "
          "ceil(roi_width / output_width), and likewise for height). Default is 0.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "mode",
          "The pooling method. Two modes are supported: 'avg' and 'max'. "
          "Default is 'avg'.",
          AttributeProto::STRING,
          std::string("avg"))
      .Input(
          0,
          "X",
          "Input data tensor from the previous operator; "
          "4-D feature map of shape (N, C, H, W), "
          "where N is the batch size, C is the number of channels, "
          "and H and W are the height and the width of the data.",
          "T1")
      .Input(
          1,
          "rois",
          "RoIs (Regions of Interest) to pool over; rois is "
          "2-D input of shape (num_rois, 4) given as "
          "[[x1, y1, x2, y2], ...]. "
          "The RoIs' coordinates are in the coordinate system of the input image. "
          "Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.",
          "T1")
      .Input(
          2,
          "batch_indices",
          "1-D tensor of shape (num_rois,) with each element denoting "
          "the index of the corresponding image in the batch.",
          "T2")
      .Output(
          0,
          "Y",
          "RoI pooled output, 4-D tensor of shape "
          "(num_rois, C, output_height, output_width). The r-th batch element Y[r-1] "
          "is a pooled feature map corresponding to the r-th RoI X[r-1].",
          "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int64)"},
          "Constrain types to int tensors.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        if (!hasNInputShapes(ctx, 3)) {
          return;
        }
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        auto& input_shape = getInputShape(ctx, 0);
        auto& rois_shape = getInputShape(ctx, 1);
        auto& batch_index_shape = getInputShape(ctx, 2);
        auto* output_shape = getOutputShape(ctx, 0);

        if (input_shape.dim_size() != 4) {
          fail_shape_inference("first input tensor has wrong dimension");
        }
        if (rois_shape.dim_size() != 2) {
          fail_shape_inference("rois input tensor has wrong dimension");
        }
        if (batch_index_shape.dim_size() != 1) {
          fail_shape_inference("batch_indices shape input tensor has wrong dimension");
        }

        output_shape->clear_dim();
        output_shape->add_dim()->set_dim_value(static_cast<int64_t>(
          rois_shape.dim(0).dim_value()));
        output_shape->add_dim()->set_dim_value(static_cast<int64_t>(
          input_shape.dim(1).dim_value()));
        output_shape->add_dim()->set_dim_value(
          ctx.getAttribute("output_height")->i());
        output_shape->add_dim()->set_dim_value(
          ctx.getAttribute("output_width")->i());
      }));

} // namespace ONNX_NAMESPACE
