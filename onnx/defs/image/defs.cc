/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* CenterCropPad_ver1_doc = R"DOC(
Center crop or pad an image to given dimensions.

The input image can have have channel-first (CHW) or channel-last layout (HWC), which can be controlled
by the `channel_first` argument.

If the input dimensions are bigger than the crop shape, a centered cropping window is extracted from the input.
If the input dimensions are smaller than the crop shape, the input is padded on each side equally,
so that the input image is centered in the output.
)DOC";

ONNX_IMAGE_OPERATOR_SET_SCHEMA(
    CenterCropPad,
    1,
    OpSchema()
        .SetDoc(CenterCropPad_ver1_doc)
        .Input(
            0,
            "input_data",
            "Input image to extract the centered crop from.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "1-D tensor representing the cropping window dimensions (height, width)",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output_data", "Output image.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr(
            "channel_first",
            "If enabled, a channel-first layout is assumed (CHW). Otherwise, a channel-last is assumed (HWC)",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_with_bfloat(),
            "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumInputs() != 2) {
            fail_type_inference("CenterCropPad op must have 2 inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if shape is initializer
          const TensorProto* cropShapeInitializer = ctx.getInputData(1);
          if (!cropShapeInitializer) {
            return;
          }

          // don't know data_type - can't proceed
          if (!cropShapeInitializer->has_data_type())
            return;

          std::vector<int64_t> shape;
          if (cropShapeInitializer->data_type() == TensorProto::INT64) {
            const auto& data = ParseData<int64_t>(cropShapeInitializer);
            shape.insert(shape.end(), data.begin(), data.end());
          } else if (cropShapeInitializer->data_type() == TensorProto::INT32) {
            const auto& data = ParseData<int32_t>(cropShapeInitializer);
            shape.insert(shape.end(), data.begin(), data.end());
          } else {
            // unaccepted data type
            fail_shape_inference("`shape` only supports `int32_t` or `int64_t` inputs");
          }

          if (shape.size() != 2) {
            fail_shape_inference("`shape` is expected to have 2 elements. Got ", shape.size(), ".");
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const int64_t input_rank = input_shape.dim_size();

          if (input_rank != 3) {
            fail_shape_inference("Input rank is expected to be 3. Got ", input_rank, ".");
          }

          auto channel_first_attr = ctx.getAttribute("channel_first");
          bool channel_first = false;
          if (channel_first_attr)
            channel_first = static_cast<bool>(channel_first_attr->i());

          int channel_dim_axis = channel_first ? 0 : 2;

          int j = 0;
          for (int i = 0; i < input_rank; ++i) {
            // first update rank of output dim
            auto* output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            const auto& input_dim = input_shape.dim(i);

            if (channel_dim_axis == i) {
              if (input_dim.has_dim_value()) {
                output_dim->set_dim_value(input_dim.dim_value());
              } else if (input_dim.has_dim_param()) {
                output_dim->set_dim_param(input_dim.dim_param());
              }
            } else {
              output_dim->set_dim_value(shape[j++]);
            }
          }
        })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
              auto channel_first_attr = ctx.getAttribute("channel_first");
              bool channel_first = false;
              if (channel_first_attr)
                channel_first = static_cast<bool>(channel_first_attr->i());

              FunctionBuilder builder(functionProto);
              builder.Const("k2", std::vector<int64_t>{2});
              builder.Add("x_shape = Shape(input_data)");

              if (channel_first) {
                builder.Const("axes", std::vector<int64_t>{1, 2});
                builder.Add("c, h, w = Split <axis = 0> (x_shape)");
              } else {
                builder.Const("axes", std::vector<int64_t>{0, 1});
                builder.Add("h, w, c = Split <axis = 0> (x_shape)");
              }

              builder.Add("hw = Concat <axis = 0> (h, w)");
              builder.Add("padded_hw = Max(hw, shape)");

              if (channel_first) {
                builder.Add("padded_sh = Concat <axis = 0> (c, padded_hw)");
              } else {
                builder.Add("padded_sh = Concat <axis = 0> (padded_hw, c)");
              }

              builder.Add("pad_amount = Sub(padded_sh, x_shape)")
                  .Add("pad_amount_left = Div(pad_amount, k2)")
                  .Add("pad_amount_right = Sub(pad_amount, pad_amount_left)")
                  .Add("pads = Concat <axis = 0> (pad_amount_left, pad_amount_right)")
                  .Add("padded_image = Pad (input_data, pads)")
                  .Add("x_shape2 = Shape(padded_image)");

              if (channel_first) {
                builder.Add("c2, h2, w2 = Split <axis = 0> (x_shape2)");
              } else {
                builder.Add("h2, w2, c2 = Split <axis = 0> (x_shape2)");
              }

              builder.Add("hw2 = Concat <axis = 0> (h2, w2)")
                  .Add("hw_diff = Sub (hw2, shape)")
                  .Add("start_xy = Div (hw_diff, k2)")
                  .Add("end_xy = Add (start_xy, shape)")
                  .Add("output_data = Slice (padded_image, start_xy, end_xy, axes)");
              schema.BuildFunction(functionProto);
              return true;
            }));

} // namespace ONNX_NAMESPACE
