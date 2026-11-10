// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_API void convPoolShapeInference(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // we need the first input shape for this inference.
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 3) {
    fail_shape_inference("Input tensor must have at least 3 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
    if (std::any_of(dilations.begin(), dilations.end(), [](int64_t d) { return d <= 0; })) {
      fail_shape_inference("Attribute dilations must only contain positive values");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides = defs::nn::utils::getConvPoolStrides(ctx, n_input_dims);

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape = ctx.getInputType(input2Idx)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
    // Reject weight/input spatial-rank mismatch; prevents OOB read of dilations/pads below.
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference(
          "Number of spatial dimensions in the weight tensor (",
          kernel_shape.size(),
          ") does not match the number of spatial dimensions in the input tensor (",
          n_input_dims,
          ").");
    }
  }

  if (std::any_of(kernel_shape.begin(), kernel_shape.end(), [](int64_t k) { return k <= 0; })) {
    fail_shape_inference("Attribute kernel_shape must only contain positive values");
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] =
        checkedAdd(checkedMultiply(checkedSubtract(effective_kernel_shape[i], 1), dilations[i]), 1);
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
    if (std::any_of(pads.begin(), pads.end(), [](int64_t p) { return p < 0; })) {
      fail_shape_inference("Attribute pads must not contain negative values");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape.dim(2 + i).has_dim_value()) {
            continue;
          }
          residual = input_shape.dim(2 + i).dim_value();
          if (residual < 0) {
            continue;
          }
          residual %= stride;
        }
        if (i >= static_cast<int>(effective_kernel_shape.size())) {
          fail_shape_inference("kernel shape should have ", input_dims_size, " values in ", ctx.getDisplayName(), ".");
        }
        int64_t total_pad = residual == 0 ? checkedSubtract(effective_kernel_shape[i], stride)
                                          : checkedSubtract(effective_kernel_shape[i], residual);
        total_pad = std::max<int64_t>(total_pad, 0);
        int64_t half_pad_small = total_pad / 2;
        int64_t half_pad_big = checkedSubtract(total_pad, half_pad_small);
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(1);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    const auto& second_input_shape = getInputShape(ctx, input2Idx);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto* newdim = output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }
    // how big is the input, including padding
    int64_t input_size = input_shape.dim(2 + i).dim_value();
    int64_t effective_input_size = checkedAdd(checkedAdd(input_size, pads[i]), pads[i + kernel_shape_size]);

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    int64_t numerator = checkedSubtract(effective_input_size, effective_kernel_shape[i]);
    if (ceil_mode) {
      numerator = checkedAdd(numerator, checkedSubtract(strides[i], 1));
    }
    int64_t output_size = checkedAdd(checkedDivide(numerator, strides[i]), 1);
    if (ceil_mode == 1 &&
        checkedMultiply(checkedSubtract(output_size, 1), strides[i]) >= checkedAdd(input_size, pads[i])) {
      // we need to match pytorch's behavior of "Sliding windows that would start in the right padded region are
      // ignored." (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#maxpool1d). this code follows the
      // same logic as PyTorch's C++ implementation:
      // https://github.com/pytorch/pytorch/blob/f1cdb39da3850c47d51ec6a5b1ae864c32b3accf/aten/src/ATen/native/Pool.h#L54C21-L54C21
      --output_size;
    }

    newdim->set_dim_value(output_size);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

static void PoolShapeInference(InferenceContext& ctx, bool use_dilation) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* output_type = ctx.getOutputType(1);
    if (output_type->value_case() == TypeProto::kTensorType || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
      output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    }
  }
  convPoolShapeInference(ctx, use_dilation, true, 0, 1);
}

static void AveragePoolShapeInference_opset22(InferenceContext& ctx) {
  PoolShapeInference(ctx, true);
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    22,
    generated::ai_onnx::AveragePool_ver22().TypeAndShapeInferenceFunction(AveragePoolShapeInference_opset22));

static void MaxPoolShapeInference_opset22(InferenceContext& ctx) {
  PoolShapeInference(ctx, true);
}

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    22,
    generated::ai_onnx::MaxPool_ver22().TypeAndShapeInferenceFunction(MaxPoolShapeInference_opset22));

static void maxUnpoolShapeInference(InferenceContext& ctx) {
  // we need at least two inputs to have a shape for this inference.
  if (ctx.getNumInputs() != 2 && ctx.getNumInputs() != 3) {
    fail_type_inference("MaxUnpool op must have either two or three inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    return; // If first input does not have shape, we cannot infer much.
  }
  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor X must have at least 2 dimensions.");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size.");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size.");
    }
  } else {
    fail_shape_inference("Attribute kernel_shape must be specified.");
  }

  if (ctx.getNumInputs() == 3) {
    // If the third input, output_size, is specified, then use that instead
    // of inferring shape from inputs.
    if (hasInputShape(ctx, 2)) {
      const auto& output_shape = getInputShape(ctx, 2);
      if (output_shape.dim_size() != 1) {
        fail_type_inference("'output_shape' must be rank 1 tensor.");
      }
      if (output_shape.dim(0).has_dim_value() &&
          static_cast<int>(output_shape.dim(0).dim_value()) != input_shape.dim_size()) {
        fail_shape_inference("'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be
            // determined at runtime.
  }

  if (!hasInputShape(ctx, 1)) {
    return; // If indices input does not have shape, we cannot infer output shape.
  }
  auto indices_shape = ctx.getInputType(1)->tensor_type().shape();
  if (indices_shape.dim_size() < 2) {
    fail_shape_inference("Indices tensor I must have at least 2 dimensions.");
  }

  auto* final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() = indices_shape.dim(1); // channels should be the second dim of second input.

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto* newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value = checkedMultiply(strides[i], checkedSubtract(input_shape.dim(2 + i).dim_value(), 1));
    newdim_value = checkedAdd(newdim_value, kernel_shape[i]);
    newdim_value = checkedSubtract(newdim_value, pads[i]);
    newdim_value = checkedSubtract(newdim_value, pads[i + kernel_shape_size]);

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    22,
    generated::ai_onnx::MaxUnpool_ver22().TypeAndShapeInferenceFunction(maxUnpoolShapeInference));

static void LpPoolShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference(ctx, true, true, 0, 1);
}

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(LpPoolShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 22, generated::ai_onnx::LpPool_ver22().FillUsing(LpPoolOpSchemaGenerator()));

// For ROI pool operations.
static void roiPoolTypeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // rois is the second input.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  // X: [N, C, H, W], rois: [num_rois, 5]
  Dim N, C, H, W, num_rois, five;
  five.set_dim_value(5);
  ctx.unifyInputShape(0, {N, C, H, W});
  ctx.unifyInputShape(1, {num_rois, five});

  std::vector<int64_t> pooled_shape;
  if (getRepeatedAttribute(ctx, "pooled_shape", pooled_shape)) {
    if (pooled_shape.size() != 2) {
      fail_shape_inference("Attribute pooled_shape has incorrect length");
    }
    for (auto dim : pooled_shape) {
      if (dim <= 0) {
        fail_shape_inference("Attribute pooled_shape must only contain positive values");
      }
    }
  } else {
    fail_shape_inference("Attribute pooled_shape must be specified");
  }

  // output: (num_rois, C, pooled_shape[0], pooled_shape[1])
  Dim pooled_h, pooled_w;
  pooled_h.set_dim_value(pooled_shape[0]);
  pooled_w.set_dim_value(pooled_shape[1]);
  updateOutputShape(ctx, 0, {num_rois, C, pooled_h, pooled_w});
}

static std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(roiPoolTypeShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(MaxRoiPool, 22, generated::ai_onnx::MaxRoiPool_ver22().FillUsing(RoiPoolOpSchemaGenerator()));

static void ConvShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference(ctx, true, false, 0, 1);
}

static std::function<void(OpSchema&)> ConvOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(ConvShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(Conv, 22, generated::ai_onnx::Conv_ver22().FillUsing(ConvOpSchemaGenerator()));

static void QLinearConvShapeInference_opset10(InferenceContext& ctx) {
  auto x_type = ctx.getInputType(0);
  auto w_type = ctx.getInputType(3);
  if (nullptr == x_type || nullptr == w_type || x_type->value_case() != TypeProto::kTensorType ||
      w_type->value_case() != TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type in ", ctx.getDisplayName(), ".");
  }

  auto x_zero_point_type = ctx.getInputType(2);
  if (nullptr == x_zero_point_type ||
      x_zero_point_type->tensor_type().elem_type() != x_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have be same type in ", ctx.getDisplayName(), ".");
  }

  auto w_zero_point_type = ctx.getInputType(5);
  if (nullptr == w_zero_point_type ||
      w_zero_point_type->tensor_type().elem_type() != w_type->tensor_type().elem_type()) {
    fail_type_inference("weight and zero_point pair is expected to have same type in ", ctx.getDisplayName(), ".");
  }

  propagateElemTypeFromInputToOutput(ctx, 7, 0);

  convPoolShapeInference(ctx, true, false, 0, 3);
}

ONNX_OPERATOR_SET_SCHEMA(
    QLinearConv,
    10,
    generated::ai_onnx::QLinearConv_ver10().TypeAndShapeInferenceFunction(QLinearConvShapeInference_opset10));

static void ConvIntegerShapeInference_opset10(InferenceContext& ctx) {
  auto x_type = ctx.getInputType(0);
  auto w_type = ctx.getInputType(1);
  auto y_type = ctx.getOutputType(0);
  if (nullptr == x_type || nullptr == w_type || nullptr == y_type || x_type->value_case() != TypeProto::kTensorType ||
      w_type->value_case() != TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
  }

  // Right now we only support int32
  y_type->mutable_tensor_type()->set_elem_type(TensorProto::INT32);

  convPoolShapeInference(ctx, true, false, 0, 1);
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvInteger,
    10,
    generated::ai_onnx::ConvInteger_ver10().TypeAndShapeInferenceFunction(ConvIntegerShapeInference_opset10));

ONNX_API void convTransposeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);
  if (group <= 0) {
    fail_shape_inference("Attribute group must be > 0 for ConvTranspose. group=", group, ".");
  }

  auto validate_input_channels_for_group = [](const TensorShapeProto& input_shape_proto, int64_t channel_group) {
    if (input_shape_proto.dim_size() < 2) {
      return;
    }

    const auto& input_channels_dim = input_shape_proto.dim(1);
    if (input_channels_dim.has_dim_value() && input_channels_dim.dim_value() % channel_group != 0) {
      fail_shape_inference(
          "Input channels C must be divisible by group for ConvTranspose. C=",
          input_channels_dim.dim_value(),
          " group=",
          channel_group,
          ".");
    }
  };

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 3) {
    fail_shape_inference(
        "Input tensor must have at least 3 dimensions (N x C x D1...Dn). Got: ", input_shape.dim_size());
  }
  validate_input_channels_for_group(input_shape, group);

  // Weight tensor (input 1) must also have at least 3 dimensions (C x M/group x k1...kn).
  auto weight_shape = ctx.getInputType(1)->tensor_type().shape();
  if (weight_shape.dim_size() < 3) {
    fail_shape_inference(
        "Weight tensor must have at least 3 dimensions (C x M/group x k1...kn). Got: ", weight_shape.dim_size());
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
    if (std::any_of(dilations.begin(), dilations.end(), [](int64_t d) { return d <= 0; })) {
      fail_shape_inference("Attribute dilations must only contain positive values");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
    if (std::any_of(strides.begin(), strides.end(), [](int64_t s) { return s <= 0; })) {
      fail_shape_inference("Attribute strides must only contain positive values");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else {
    for (int i = 2; i < weight_shape.dim_size(); ++i) {
      if (!weight_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(weight_shape.dim(i).dim_value());
    }
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Weight tensor spatial rank does not match input tensor spatial rank");
    }
  }

  if (std::any_of(kernel_shape.begin(), kernel_shape.end(), [](int64_t k) { return k <= 0; })) {
    fail_shape_inference("Attribute kernel_shape must only contain positive values");
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] =
        checkedAdd(checkedMultiply(checkedSubtract(effective_kernel_shape[i], 1), dilations[i]), 1);
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
    if (std::any_of(pads.begin(), pads.end(), [](int64_t p) { return p < 0; })) {
      fail_shape_inference("Attribute pads must not contain negative values");
    }
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if (nullptr != auto_pad_attr && auto_pad_attr->s() != "NOTSET") {
      fail_shape_inference("The pads attribute cannot be used simultaneously with auto_pad attribute");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t total_pad = checkedSubtract(effective_kernel_shape[i], strides[i]);
        total_pad = std::max<int64_t>(total_pad, 0);
        int64_t half_pad_small = total_pad / 2;
        int64_t half_pad_big = checkedSubtract(total_pad, half_pad_small);
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute output_shape has incorrect size");
    }
    if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t s) { return s < 0; })) {
      fail_shape_inference("Attribute output_shape must not contain negative values");
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) { // Added only to one side.
      fail_shape_inference("Attribute output_padding has incorrect size");
    }
    if (std::any_of(output_padding.begin(), output_padding.end(), [](int64_t p) { return p < 0; })) {
      fail_shape_inference("Attribute output_padding must not contain negative values");
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto* final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() = weight_shape.dim(1) * group; // channels should be the second dim of second input
                                                                // multiply group.

  int size_of_output = 0;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value() && output_shape[i] < input_shape.dim(i + 2).dim_value()) {
        // TODO(ONNX): throw exception?
        return; // output shape value cannot be smaller than the input shape
                // value
      }

      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim = checkedMultiply(strides[i], checkedSubtract(input_shape.dim(i + 2).dim_value(), 1));
        output_shape_dim = checkedAdd(output_shape_dim, output_padding[i]);
        output_shape_dim = checkedAdd(output_shape_dim, effective_kernel_shape[i]);
        output_shape_dim = checkedSubtract(output_shape_dim, pads[i]);
        output_shape_dim = checkedSubtract(output_shape_dim, pads[i + n_input_dims]);
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(convTransposeShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    22,
    generated::ai_onnx::ConvTranspose_ver22().FillUsing(ConvTransposeOpSchemaGenerator()));

static void DeformConvShapeInference_opset22(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference(ctx, true, false, 0, 1);
}

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    22,
    generated::ai_onnx::DeformConv_ver22().TypeAndShapeInferenceFunction(DeformConvShapeInference_opset22));

// For GlobalPool operations.
ONNX_API void globalPoolTypeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // X: [N, C, D1, ..., Dn] -> Y: [N, C, 1, 1, ..., 1]
  Dim N, C;
  ctx.unifyInputShapePrefix(0, {N, C});

  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = N;
  *output_shape->add_dim() = C;

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

static std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(globalPoolTypeShapeInference); };
}
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    22,
    generated::ai_onnx::GlobalAveragePool_ver22().FillUsing(GlobalPoolingOpSchemaGenerator()));
ONNX_OPERATOR_SET_SCHEMA(
    GlobalMaxPool,
    22,
    generated::ai_onnx::GlobalMaxPool_ver22().FillUsing(GlobalPoolingOpSchemaGenerator()));

static std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(globalPoolTypeShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    22,
    generated::ai_onnx::GlobalLpPool_ver22().FillUsing(GlobalLpPoolingOpSchemaGenerator()));

static void BatchNormalizationShapeInference_opset15(InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
  propagateShapeFromInputToOutput(ctx, 0, 0);

  // Inputs 1 to 4 must be of rank 1.
  checkInputRank(ctx, 1, 1);
  checkInputRank(ctx, 2, 1);
  checkInputRank(ctx, 3, 1);
  checkInputRank(ctx, 4, 1);

  Dim num_channels;

  if (hasInputShape(ctx, 0)) {
    if (getInputShape(ctx, 0).dim_size() > 1)
      unifyInputDim(ctx, 0, 1, num_channels);
    else
      unifyDim(num_channels, 1);
  }

  unifyInputDim(ctx, 1, 0, num_channels);
  unifyInputDim(ctx, 2, 0, num_channels);
  unifyInputDim(ctx, 3, 0, num_channels);
  unifyInputDim(ctx, 4, 0, num_channels);

  if (ctx.getAttribute("training_mode") && static_cast<int>(ctx.getAttribute("training_mode")->i()) != 0) {
    if (ctx.getNumOutputs() != 3)
      fail_shape_inference("This number of op outputs should be 3 when Training_mode = True, but it is not.");
  } else {
    if (ctx.getNumOutputs() != 1)
      fail_shape_inference("This number of op outputs should be 1 when Training_mode = False, but it is not.");
  }

  if (ctx.getNumOutputs() > 1) {
    TensorShapeProto outputs_shape;
    *outputs_shape.add_dim() = num_channels; // channel

    propagateElemTypeFromInputToOutput(ctx, 3, 1);
    updateOutputShape(ctx, 1, outputs_shape);

    if (ctx.getNumOutputs() > 2) {
      propagateElemTypeFromInputToOutput(ctx, 4, 2);
      updateOutputShape(ctx, 2, outputs_shape);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    15,
    generated::ai_onnx::BatchNormalization_ver15().TypeAndShapeInferenceFunction(
        BatchNormalizationShapeInference_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    22,
    generated::ai_onnx::InstanceNormalization_ver22().TypeAndShapeInferenceFunction(
        propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    22,
    generated::ai_onnx::LpNormalization_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void DropoutShapeInference_opset22(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }

  if (ctx.getNumInputs() > 1 && hasInputShape(ctx, 1)) {
    auto& ratio_input_shape = getInputShape(ctx, 1);
    if (static_cast<int>(ratio_input_shape.dim_size()) != 0) {
      fail_shape_inference("Ratio of Dropout must be a scalar.");
    }
  }

  if (ctx.getNumInputs() > 2 && hasInputShape(ctx, 2)) {
    auto& training_mode_input_shape = getInputShape(ctx, 2);
    if (static_cast<int>(training_mode_input_shape.dim_size()) != 0) {
      fail_shape_inference("training_mode of Dropout must be a scalar.");
    }
  }

  if (ctx.getNumOutputs() == 2) {
    updateOutputElemType(ctx, 1, TensorProto::BOOL);
    if (hasNInputShapes(ctx, 1)) {
      propagateShapeFromInputToOutput(ctx, 0, 1);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    22,
    generated::ai_onnx::Dropout_ver22().TypeAndShapeInferenceFunction(DropoutShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    Shrink,
    9,
    generated::ai_onnx::Shrink_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void FlattenShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0))
    return;
  auto& input_shape = getInputShape(ctx, 0);
  int rank = static_cast<int>(input_shape.dim_size());
  int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
  if (axis < 0) {
    axis += rank;
  }
  if (axis > rank || axis < 0) {
    fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
  }
  // TODO(ONNX): is the operation defined for input-rank < 2?
  updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
}

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    25,
    generated::ai_onnx::Flatten_ver25().TypeAndShapeInferenceFunction(FlattenShapeInference_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    13,
    generated::ai_onnx::LRN_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void TfIdfVectorizerShapeInference_opset9(InferenceContext& ctx) {
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_elem_type->set_elem_type(TensorProto::FLOAT);

  if (hasInputShape(ctx, 0)) {
    std::vector<int64_t> ngram_indexes;
    getRepeatedAttribute(ctx, "ngram_indexes", ngram_indexes);
    if (ngram_indexes.empty() ||
        !std::all_of(ngram_indexes.cbegin(), ngram_indexes.cend(), [](int64_t i) { return i >= 0; })) {
      fail_shape_inference("ngram_indexes must be non-empty with no negative values");
    }

    auto greatest_hit = std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend());
    auto max_last_axis = *greatest_hit + 1;

    TensorShapeProto output_shape;
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    auto dim_size = input_shape.dim_size();
    if (dim_size == 1) {
      output_shape.add_dim()->set_dim_value(max_last_axis);
    } else if (dim_size == 2) {
      *output_shape.add_dim() = input_shape.dim(0);
      output_shape.add_dim()->set_dim_value(max_last_axis);
    } else {
      fail_shape_inference("Input tensor must have rank 1 or 2");
    }
    updateOutputShape(ctx, 0, output_shape);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    TfIdfVectorizer,
    9,
    generated::ai_onnx::TfIdfVectorizer_ver9().TypeAndShapeInferenceFunction(TfIdfVectorizerShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(MeanVarianceNormalization, 13, generated::ai_onnx::MeanVarianceNormalization_ver13());

static void col2imShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // All inputs shapes are required
  if (!hasNInputShapes(ctx, 3)) {
    return;
  }

  // We assume image_shape has correct spatial dimensions for next validations
  // An alternative is get the the number of spatial dimensions as an input argument
  Dim n_input_dims;
  unifyInputDim(ctx, 1, 0, n_input_dims);

  unifyInputDim(ctx, 2, 0, n_input_dims);
  checkInputRank(ctx, 1, 1);
  checkInputRank(ctx, 2, 1);
  std::vector<int64_t> image_shape = {};
  const TensorProto* image_shape_data = ctx.getInputData(1);
  if (image_shape_data) {
    image_shape = ParseData<int64_t>(image_shape_data);
    unifyDim(n_input_dims, static_cast<int64_t>(image_shape.size()));
  }

  std::vector<int64_t> pads = {};
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() % 2) {
      fail_shape_inference("Attribute pads must have an even size");
    }
    unifyDim(n_input_dims, static_cast<int64_t>(pads.size() / 2));
  }

  std::vector<int64_t> dilations = {};
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    unifyDim(n_input_dims, static_cast<int64_t>(dilations.size()));
  }

  std::vector<int64_t> strides = {};
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    unifyDim(n_input_dims, static_cast<int64_t>(strides.size()));
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() != 3) {
    fail_shape_inference("input must have rank 3.");
  }

  std::vector<int64_t> block_shape = {};
  const TensorProto* block_shape_data = ctx.getInputData(2);
  if (block_shape_data) {
    block_shape = ParseData<int64_t>(block_shape_data);
    unifyDim(n_input_dims, static_cast<int64_t>(block_shape.size()));
  }
  unifyInputDim(ctx, 2, 0, n_input_dims);

  int64_t block_shape_size = 0;
  if (!block_shape.empty()) {
    block_shape_size = 1;
    for (const auto& dim : block_shape) {
      if (dim <= 0) {
        fail_shape_inference("'block_shape' input must contain only positive values");
      }
      block_shape_size = checkedMultiply(block_shape_size, dim);
    }
  }
  // If we haven't inferred the number of image dimensions, we can't set inferred shape.
  if (!n_input_dims.has_dim_value()) {
    return;
  }
  // Do not materialize an attacker-controlled number of Dimension messages
  // when only Col2Im's spatial rank is known.
  if (n_input_dims.dim_value() < 0 || n_input_dims.dim_value() > kMaxMaterializedRank) {
    return;
  }

  // Final shape will be (N, C, dim_1, ..., dim_N)
  auto* final_image_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  // Dimensions N and C are always present
  Dim N, C;
  if (ctx.getInputType(0)->tensor_type().shape().dim(0).has_dim_value()) {
    N = input_shape.dim(0); // Otherwise, N is unknown.
  }
  *final_image_shape->add_dim() = N;

  if (block_shape_size > 0) {
    C = input_shape.dim(1) / block_shape_size; // Otherwise, C is unknown.
  }
  *final_image_shape->add_dim() = C;

  // Image dimensions are dynamic
  for (auto i = 0; i < n_input_dims.dim_value(); ++i) {
    Dim image_dim_i;
    if (!image_shape.empty()) {
      image_dim_i.set_dim_value(image_shape[i]); // Otherwise, spatial dimensions are unknown
    }
    *final_image_shape->add_dim() = image_dim_i;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Col2Im,
    18,
    generated::ai_onnx::Col2Im_ver18().TypeAndShapeInferenceFunction(col2imShapeInference));

static bool BuildContextDependentFunctionBodyLayerNormalization(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto,
    int sinceVersion) {
  ONNX_ASSERT(sinceVersion == 17 || sinceVersion == 18)
  // LayerNormalization <axis, epsilon, stash_type> (X, Scale, B) => (Y, Mean?, InvStdDev?)
  const auto* const tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  int64_t T = tp->tensor_type().elem_type();

  const auto* const type_attr = ctx.getAttribute("stash_type");
  int64_t U =
      (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16))
    return false; // Error

  const auto* const axis_attr = ctx.getAttribute("axis");
  int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
  const auto* const epsilon_attr = ctx.getAttribute("epsilon");
  float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

  auto mktensor = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
    auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
    tp.add_dims(1);
    return tp;
  };
  // The treatment of "axis" is different in "LayerNormalization" and in Reduction operations.
  // This complicates the function definition, requiring reshaping inputs/outputs.
  // Input X shape: [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]
  // This is treated as a 2D shape [d[0] * ... * d[axis-1], d[axis] * ... * d[rank-1]]
  // Normalization is applied to the second dimension.
  // Output Y has same shape as X
  // Outputs Mean and InvStdDev have shape: [d[0], ..., d[axis-1], 1, ..., 1]
  FunctionBuilder builder(functionProto);
  builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
      .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
      .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
      .Add("Rank = Size (XShape)") // rank of input tensor: scalar
      .Add("Zero1D = Constant()", "value", mktensor(0)) // [0] : 1D tensor
      .Add("Axis1D = Constant()", "value", mktensor(axis)) // [axis] : 1D tensor
      .Add("PrefixShape = Slice (XShape, Zero1D, Axis1D)") // [d[0], ..., d[axis-1]]
      .Add(
          axis >= 0 // number of axes that are reduced =
              ? "NumReducedAxes = Sub (Rank, Axis1D)" // [rank - axis]: 1D tensor
              : "NumReducedAxes = Neg (Axis1D)") // [-axis] : 1D tensor
      .Add(
          "SuffixShape = ConstantOfShape (NumReducedAxes)",
          "value",
          mktensor(1)) // [1, ..., 1] for reduced axes
      .Add("ReducedShape = Concat <axis = 0> (PrefixShape, SuffixShape)") // [d[0], ..., d[axis-1], 1, ..., 1]
      .Add("X2D = Flatten (X)", "axis", axis)
      .Add("XU = Cast (X2D)", "to", U);
  if (sinceVersion == 17) {
    builder.Add("Mean2D = ReduceMean <axes = [1]> (XU)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean <axes = [1]> (Square)");
  } else if (sinceVersion == 18) {
    builder.Add("Axes_1 = Constant()", "value", mktensor(1))
        .Add("Mean2D = ReduceMean (XU, Axes_1)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean (Square, Axes_1)");
  }
  builder.Add("SquareOfMean = Mul (Mean2D, Mean2D)")
      .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
      .Add("VarPlusEpsilon = Add (Var, Epsilon)")
      .Add("StdDev = Sqrt (VarPlusEpsilon)")
      .Add("Deviation = Sub (XU, Mean2D)")
      .Add("Normalized = Div (Deviation, StdDev)")
      .Add("NormalizedT = Cast (Normalized)", "to", T)
      .Add("Scale2D = Flatten <axis = 0> (Scale)")
      .Add("Scaled = Mul (NormalizedT, Scale2D)");
  if (ctx.hasInput(2)) {
    builder.Add("B2D = Flatten <axis=0> (B)");
    builder.Add("Biased = Add (Scaled, B2D)");
  } else {
    builder.Add("Biased = Identity (Scaled)");
  }
  builder.Add("Y = Reshape (Biased, XShape)");
  builder.Add("InvStdDev2D = Reciprocal (StdDev)");
  if (ctx.hasOutput(1))
    builder.Add("Mean = Reshape (Mean2D, ReducedShape)");
  if (ctx.hasOutput(2))
    builder.Add("InvStdDev = Reshape (InvStdDev2D, ReducedShape)");

  schema.BuildFunction(functionProto);
  return true;
}

static bool BuildContextDependentFunctionBodyLayerNormalizationVer17(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 17);
}

static bool BuildContextDependentFunctionBodyLayerNormalizationVer18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 18);
}

static void LayerNormalizationShapeInference_opset17(InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
  auto stash_type = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto stash_type_proto = ctx.getAttribute("stash_type");
  if (stash_type_proto) {
    stash_type = stash_type_proto->i();
  }
  if (ctx.getNumOutputs() > 1) {
    auto output_type = ctx.getOutputType(1);
    output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
  }
  if (ctx.getNumOutputs() > 2) {
    auto output_type = ctx.getOutputType(2);
    output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto& input_shape = getInputShape(ctx, 0);
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = -1;
  auto axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
  }
  if (axis < 0) {
    // Convert negative axis value to equivalent
    // positive value.
    axis += input_ndim;
  }
  if (axis < 0 || axis >= input_ndim) {
    fail_shape_inference(
        "Unexpected axis value (", axis, ") rank of first input is ", input_ndim, " in ", ctx.getDisplayName(), ".");
  }
  if (ctx.getNumOutputs() > 1) {
    auto mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    mean_shape->CopyFrom(input_shape);
    for (auto d = axis; d < input_ndim; ++d)
      mean_shape->mutable_dim(d)->set_dim_value(1);
  }

  if (ctx.getNumOutputs() > 2) {
    auto inv_std_dev_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
    inv_std_dev_shape->CopyFrom(input_shape);
    for (auto d = axis; d < input_ndim; ++d)
      inv_std_dev_shape->mutable_dim(d)->set_dim_value(1);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    LayerNormalization,
    17,
    generated::ai_onnx::LayerNormalization_ver17()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer17, 17)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer18, 18)
        .TypeAndShapeInferenceFunction(LayerNormalizationShapeInference_opset17));

static void GroupNormalizationShapeInference(InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);

  const int64_t num_groups = getAttribute(ctx, "num_groups", 0);
  if (num_groups <= 0) {
    fail_shape_inference("Attribute num_groups must be > 0. num_groups=", num_groups, ".");
  }

  if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() < 2) {
    fail_shape_inference("Input X must have rank >= 2.");
  }

  checkInputRank(ctx, 1, 1);
  checkInputRank(ctx, 2, 1);

  Dim num_channels;
  unifyInputDim(ctx, 0, 1, num_channels);
  unifyInputDim(ctx, 1, 0, num_channels);
  unifyInputDim(ctx, 2, 0, num_channels);

  if (num_channels.has_dim_value() && num_channels.dim_value() % num_groups != 0) {
    fail_shape_inference(
        "The number of channels must be divisible by num_groups. channels=",
        num_channels.dim_value(),
        " num_groups=",
        num_groups,
        ".");
  }
}
static bool GroupNormalizationFunctionBody_opset21(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // GroupNormalization <epsilon, num_groups> (X, scale, bias) => (Y)
  auto tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  int64_t in_type = tp->tensor_type().elem_type();

  auto epsilon_attr = ctx.getAttribute("epsilon");
  float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;
  auto num_groups_attr = ctx.getAttribute("num_groups");
  if (num_groups_attr == nullptr)
    return false;
  int64_t num_groups = num_groups_attr->i();

  auto stash_type_attr = ctx.getAttribute("stash_type");
  int64_t stash_type = (stash_type_attr != nullptr) ? stash_type_attr->i()
                                                    : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if ((stash_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
      (stash_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
      (stash_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
      (stash_type != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
    return false; // Error

  FunctionBuilder builder(functionProto);
  builder.Const1D("FloatEpsilon", epsilon)
      .Add("Epsilon = Cast (FloatEpsilon)", "to", stash_type)
      .Add("XU = Cast (X)", "to", stash_type)
      .Add("XShape = Shape (XU)") // shape of input tensor: 1D tensor
      .Add("C = Shape <start = 1, end = 2> (X)")
      .Const1D("NumGroups", num_groups)
      .Add("GroupSize = Div (C, NumGroups)")
      .Add("N = Shape <start = 0, end = 1> (X)") // batch size
      .Add("InstanceShape = Shape <start = 2> (X)") // data instance shape

      // NewShape = [N, num_groups, group_size, H, W, (...)]
      .Add("NewShape = Concat <axis = 0> (N, NumGroups, GroupSize, InstanceShape)")
      .Add("XReshaped = Reshape (XU, NewShape)")

      // Flatten into 3D tensor: [N, num_groups, group_size x H x W (x ...)]
      .Add("Shape3D = Constant <value_ints = [0, 0, -1]> ()")
      .Add("X3D = Reshape (XReshaped, Shape3D)")

      // Calculate statistics
      .Const1D("Axes2", static_cast<int64_t>(2))
      .Add("Mean = ReduceMean (X3D, Axes2)")
      .Add("Square = Mul (X3D, X3D)")
      .Add("MeanOfSquare = ReduceMean (Square, Axes2)")
      .Add("SquareOfMean = Mul (Mean, Mean)")
      .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
      .Add("VarPlusEpsilon = Add (Var, Epsilon)")
      .Add("StdDev = Sqrt (VarPlusEpsilon)")
      .Add("Deviation = Sub (X3D, Mean)")
      .Add("NormalizedU = Div (Deviation, StdDev)")

      // Reshape to [N, C, H x W (x ...)] and cast to original type
      .Add("NormalizedOriginalShape = Reshape (NormalizedU, XShape)")
      .Add("NormalizedNC = Reshape (NormalizedOriginalShape, Shape3D)")
      .Add("NormalizedT = Cast (NormalizedNC)", "to", in_type)

      // Reshape scale and bias to [1, C, 1] for broadcasting
      .Add("ScaleShape = Constant <value_ints = [1, -1, 1]> ()")
      .Add("ScaleT = Cast (scale)", "to", in_type)
      .Add("BiasT = Cast (bias)", "to", in_type)
      .Add("ScaleReshaped = Reshape (ScaleT, ScaleShape)")
      .Add("BiasReshaped = Reshape (BiasT, ScaleShape)")

      // Calculate scaled and biased output
      .Add("Scaled = Mul (ScaleReshaped, NormalizedT)")
      .Add("Biased = Add (Scaled, BiasReshaped)")
      .Add("Y = Reshape (Biased, XShape)");

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    21,
    generated::ai_onnx::GroupNormalization_ver21()
        .TypeAndShapeInferenceFunction(GroupNormalizationShapeInference)
        .SetContextDependentFunctionBodyBuilder(GroupNormalizationFunctionBody_opset21));

static bool RMSNormalizationFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // RMSNormalization <axis, epsilon, stash_type> (X, Scale) => (Y)
  auto tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  int64_t T = tp->tensor_type().elem_type();

  auto type_attr = ctx.getAttribute("stash_type");
  int64_t U =
      (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
      (U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) && (U != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
    return false; // Error

  auto axis_attr = ctx.getAttribute("axis");
  int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
  auto epsilon_attr = ctx.getAttribute("epsilon");
  float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

  FunctionBuilder builder(functionProto);
  builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
      .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
      .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
      .Add("Rank = Size (XShape)") // rank of input tensor: scalar
      .Const("Axis", axis) // axis : scalar
      .Add(
          axis >= 0 // number of axes that are reduced =
              ? "PosAxis = Identity (Axis)" // axis: scalar
              : "PosAxis = Add (Rank, Axis)") // rank + axis : scalar
      .Const("One", static_cast<int64_t>(1))
      .Add("ReduceAxes = Range(PosAxis, Rank, One)")
      .Add("XU = Cast (X)", "to", U);
  builder.Add("XSquared = Mul (XU, XU)")
      .Add("XSquaredMean = ReduceMean (XSquared, ReduceAxes)")
      .Add("MeanSquareEpsilon = Add (XSquaredMean, Epsilon)")
      .Add("RMS = Sqrt (MeanSquareEpsilon)")
      .Add("Normalized = Div (XU, RMS)")
      .Add("NormalizedT = Cast (Normalized)", "to", T);
  builder.Add("Y = Mul (NormalizedT, scale)");

  schema.BuildFunction(functionProto);
  return true;
}

static void RMSNormalizationShapeInference_opset23(ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = -1;
  auto axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
  }
  if (axis < 0) {
    // Convert negative axis value to equivalent
    // positive value.
    axis += input_ndim;
  }
  if (axis < 0) {
    fail_shape_inference(
        "Unexpected axis value (", axis, ") rank of first input is ", input_ndim, " in ", ctx.getDisplayName(), ".");
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    RMSNormalization,
    23,
    generated::ai_onnx::RMSNormalization_ver23()
        .TypeAndShapeInferenceFunction(RMSNormalizationShapeInference_opset23)
        .SetContextDependentFunctionBodyBuilder(RMSNormalizationFunctionBody_opset23));

static bool RotaryEmbeddingFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // RotaryEmbedding <scale, interleaved, rotary_embedding_dim, num_heads> (X, position_ids, cos_cache,
  // sin_cache) => Y

  int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  auto interleaved_attr = ctx.getAttribute("interleaved");
  int64_t interleaved = (interleaved_attr != nullptr) ? interleaved_attr->i() : 0;
  auto rotary_embedding_dim_attr = ctx.getAttribute("rotary_embedding_dim");
  int64_t rotary_embedding_dim = (rotary_embedding_dim_attr != nullptr) ? rotary_embedding_dim_attr->i() : 0;
  auto num_heads_attr = ctx.getAttribute("num_heads");
  int64_t num_heads = (num_heads_attr != nullptr) ? num_heads_attr->i() : 0;

  // Ensure that num_heads does not control reshaping of input tensor
  // when input tensor is 4D
  int64_t is_input_4d = 1;
  auto x_tp = ctx.getInputType(0);
  if ((x_tp == nullptr) || (!x_tp->has_tensor_type()))
    return false;
  if (!(x_tp->tensor_type().has_shape())) {
    return false;
  }
  if (x_tp->tensor_type().shape().dim_size() == 4) {
    is_input_4d = 1;
  } else if (x_tp->tensor_type().shape().dim_size() == 3) {
    is_input_4d = 0;
  } else {
    return false;
  }

  FunctionBuilder builder(functionProto);
  // Set input tensor to the correct shape if input shape is 3D
  // NewShape = [batch_size, sequence_length, num_heads, head_size]

  // Reshape tensor to 4D if input is 3D
  builder.Const1D("Zero1D", static_cast<int64_t>(0))
      .Const1D("NumHeads", num_heads) // num_heads
      .Const1D("NegOne", static_cast<int64_t>(-1)); // head_size, inferred from other dimensions

  if (is_input_4d == 0) {
    builder.Add("NewShape = Concat <axis = 0> (Zero1D, Zero1D, NumHeads, NegOne)")
        .Add("XIn = Reshape (X, NewShape)"); // new shape of input tensor: 4D tensor
  } else {
    builder.Add("XIn = Transpose <perm = [0, 2, 1, 3]> (X)");
  }

  // Rotary embedding dimension is the value along which the input is to be split
  // There are two cases for the rotary embedding dimension:
  // 1. Complete rotation: rotary embedding dimension defaults to head_size, rotary_embedding_dim = cos.shape[3]
  // * 2 or head_size
  // 2. Partial rotation: rotary embedding dimension is provided, rotary_embedding_dim = rotary_embedding_dim

  builder.Add("HeadSize = Shape <start = 3, end = 4> (XIn)");
  if (rotary_embedding_dim > 0) {
    builder.Const1D("RotaryEmbedDim", rotary_embedding_dim);
  } else {
    builder.Add("RotaryEmbedDim = Identity(HeadSize)");
  }
  builder.Const1D("Two1D", static_cast<int64_t>(2))
      .Add("NoRotateLength = Sub(HeadSize, RotaryEmbedDim)")
      .Add("RotateSplitLengths = Concat <axis = 0> (RotaryEmbedDim, NoRotateLength)");
  // shape of input to rotate = input[:,:,:,:rotary_embedding_dim]
  // shape of input not to rotate = input[:,:,:,rotary_embedding_dim:]
  builder.Add("XToRotate, XNoRotate = Split <axis = -1> (XIn, RotateSplitLengths)");

  // Gather the cos and sine matrices from the respective caches using position ids if provided.
  // Otherwise Gather op functions as an Identity op.
  // Unsqueeze applied to make cos and sin matrices have dimensions that are
  // valid for multiplication with input when is split. For cases where rotary_embedding_dim is provided,
  // slice the matrix values until that index only
  if (ctx.hasInput(3)) {
    builder
        .Add("CosCacheGather = Gather(cos_cache, position_ids)") // shape of cos matrix: [batch_size,
                                                                 // sequence_length, head_size / 2]
        .Add("SinCacheGather = Gather(sin_cache, position_ids)"); // shape of cos matrix: [batch_size,
                                                                  // sequence_length, head_size / 2]
  } else {
    builder
        .Add("CosCacheGather = Identity(cos_cache)") // shape of cos matrix: [batch_size, sequence_length,
                                                     // head_size / 2]
        .Add("SinCacheGather = Identity(sin_cache)"); // shape of cos matrix: [batch_size, sequence_length,
                                                      // head_size / 2]
  }

  builder.Add("RotaryEmbedDimHalf = Div(RotaryEmbedDim, Two1D)")
      .Add("RotaryEmbedDimHalfInt = Cast (RotaryEmbedDimHalf)", "to", int_type)
      .Add("CosCacheSliced = Slice(CosCacheGather, Zero1D, RotaryEmbedDimHalfInt, Two1D)") // shape of cos
                                                                                           // matrix:
                                                                                           // [batch_size,
                                                                                           // sequence_length,
                                                                                           // rotary_embedding_dim
                                                                                           // / 2]
      .Add("SinCacheSliced = Slice(SinCacheGather, Zero1D, RotaryEmbedDimHalfInt, Two1D)") // shape of sin
                                                                                           // matrix:
                                                                                           // [batch_size,
                                                                                           // sequence_length,
                                                                                           // rotary_embedding_dim
                                                                                           // / 2]
      .Add("CosCacheUnsqueezed = Unsqueeze(CosCacheSliced, Two1D)") // shape of cos matrix: [batch_size,
                                                                    // sequence_length, 1, rotary_embedding_dim
                                                                    // / 2]
      .Add("SinCacheUnsqueezed = Unsqueeze(SinCacheSliced, Two1D)"); // shape of sin matrix: [batch_size,
                                                                     // sequence_length, 1, rotary_embedding_dim
                                                                     // / 2]

  // Create slices of inputs to multiply with sin and cos matrices based on interleaved parameter
  // Choose the correct slices based on interleaved parameter
  // real = cos_x * x1 - sin_x * x2
  // imag = sin_x * x1 + cos_x * x2
  if (interleaved == 0) {
    // For non-interleaved (basic) rotation, slices are created as follows,
    builder.Add(
        "X1, X2 = Split <axis = -1, num_outputs = 2> (XToRotate)"); // shape of X1 =
                                                                    // input[:,:,:,:rotary_embedding_dim/2],
                                                                    // X2 =
                                                                    // input[:,:,:,rotary_embedding_dim/2:rotary_embedding_dim]
  } else {
    // For interleaved rotation, slices are created as follows,
    builder.Const1D("One1D", static_cast<int64_t>(1))
        .Const1D("AxesRotaryDim", static_cast<int64_t>(3))
        .Add("RotaryEmbedDimInclusive = Add(RotaryEmbedDim, One1D)")
        .Add(
            "X1 = Slice(XToRotate, Zero1D, RotaryEmbedDim, AxesRotaryDim, Two1D)") // shape of X1 =
                                                                                   // input[:,:,:,0:rotary_embedding_dim:2]
        .Add(
            "X2 = Slice(XToRotate, One1D, RotaryEmbedDimInclusive, AxesRotaryDim, Two1D)"); // shape of
                                                                                            // X2 =
                                                                                            // input[:,:,:,1:rotary_embedding_dim:2]
  }

  builder.Add("CosX1 = Mul(CosCacheUnsqueezed, X1)")
      .Add("SinX2 = Mul(SinCacheUnsqueezed, X2)")
      .Add("Real = Sub(CosX1, SinX2)")
      .Add("SinX1 = Mul(SinCacheUnsqueezed, X1)")
      .Add("CosX2 = Mul(CosCacheUnsqueezed, X2)")
      .Add("Imaginary = Add(SinX1, CosX2)");

  // Insert the real and imaginary values into the original input to be rotated based on interleaved parameter
  if (interleaved == 0) {
    builder.Add("XRotated = Concat <axis = -1> (Real, Imaginary)");
  } else {
    builder
        .Add("RealInterleave = Unsqueeze(Real, NegOne)") // shape of indices =
                                                         // input[:,:,:,0:rotary_embedding_dim:2, 1]
        .Add("ImaginaryInterleave = Unsqueeze(Imaginary, NegOne)") // shape of indices =
                                                                   // input[:,:,:,1:rotary_embedding_dim+1:2, 1]
        .Add("XRotatedInterleavedConcat = Concat <axis = -1> (RealInterleave, ImaginaryInterleave)")
        .Add("XRotatedShape = Shape(XToRotate)")
        .Add("XRotated = Reshape(XRotatedInterleavedConcat, XRotatedShape)");
  }

  // Combine rotated parts with non-rotated parts
  builder.Add("XConcat = Concat <axis = -1> (XRotated, XNoRotate)");

  if (is_input_4d == 0) {
    builder.Add("YTransposed = Identity(XConcat)");
  } else {
    builder.Add("YTransposed = Transpose <perm = [0, 2, 1, 3]> (XConcat)");
  }
  // Reshape back to 3D shape if input is a 3D tensor
  builder.Add("XShape = Shape(X)").Add("Y = Reshape(YTransposed, XShape)");

  schema.BuildFunction(functionProto);
  return true;
}

static void RotaryEmbeddingShapeInference_opset23(ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  propagateShapeFromInputToOutput(ctx, 0, 0);

  // we need at least one input to have a shape for this inference.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if ((input_shape.dim_size() < 3) || (input_shape.dim_size() > 4)) {
    fail_shape_inference("Input tensor must have at least 3 and at most 4 dimensions");
  }

  auto num_heads_attr = ctx.getAttribute("num_heads");
  if ((input_shape.dim_size() == 3) && (num_heads_attr == nullptr)) {
    fail_shape_inference("Input shape is 3D, num_heads attribute must be provided");
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    RotaryEmbedding,
    23,
    generated::ai_onnx::RotaryEmbedding_ver23()
        .TypeAndShapeInferenceFunction(RotaryEmbeddingShapeInference_opset23)
        .SetContextDependentFunctionBodyBuilder(RotaryEmbeddingFunctionBody_opset23));

static void Attention25Inference(InferenceContext& ctx) {
  defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput(ctx);

  for (const char* window_attr_name : {"left_window_size", "right_window_size"}) {
    const auto* const window_attr = ctx.getAttribute(window_attr_name);
    const int64_t window_size = window_attr != nullptr ? window_attr->i() : -1;
    if (window_size < -1) {
      fail_shape_inference(window_attr_name, " must be -1 or nonnegative, got ", window_size);
    }
  }

  if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() == 3) {
    const auto* const q_num_heads_attr = ctx.getAttribute("q_num_heads");
    const auto* const kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
    if (q_num_heads_attr != nullptr && kv_num_heads_attr != nullptr &&
        (q_num_heads_attr->i() <= 0 || kv_num_heads_attr->i() <= 0)) {
      fail_shape_inference("q_num_heads and kv_num_heads must be positive for 3D inputs.");
    }
  }

  const bool has_past_key = ctx.hasInput(4);
  const bool has_past_value = ctx.hasInput(5);
  if (has_past_key != has_past_value) {
    fail_shape_inference("past_key and past_value must be provided together.");
  }
  const bool has_present_key = ctx.hasOutput(1);
  const bool has_present_value = ctx.hasOutput(2);
  if (has_present_key != has_present_value) {
    fail_shape_inference("present_key and present_value must be requested together.");
  }
  if (ctx.hasInput(6) && (has_past_key || has_present_key)) {
    fail_shape_inference("nonpad_kv_seqlen cannot be combined with past or present cache tensors.");
  }

  if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() == 4 &&
      (ctx.getAttribute("q_num_heads") != nullptr || ctx.getAttribute("kv_num_heads") != nullptr)) {
    fail_shape_inference("q_num_heads and kv_num_heads must not be specified for 4D inputs.");
  }
}

static bool AttentionFunctionBody_opset25(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // ScaledDotProductAttention <scale, is_causal, q_num_heads, kv_numheads, left_window_size,
  // right_window_size> (Q, K, V,
  // attn_mask, past_key, past_value) => (Y, present_key?, present_value?)
  int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  int64_t float_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

  // Get input types
  auto t_qk = ctx.getInputType(0);
  if ((t_qk == nullptr) || (!t_qk->has_tensor_type()))
    return false;
  int64_t T1 = t_qk->tensor_type().elem_type();

  // Determine precision types for Softmax
  auto softmax_precision_attr = ctx.getAttribute("softmax_precision");
  int64_t softmax_precision = (softmax_precision_attr != nullptr) ? softmax_precision_attr->i() : T1;
  if ((softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
      (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
      (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
      (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
    return false; // Error

  // If shape is 3D, q_num_heads and kv_num_heads is provided,
  // for 4D cases, set num_heads to zero for reshape purposes
  auto q_num_heads_attr = ctx.getAttribute("q_num_heads");
  int64_t q_num_heads = (q_num_heads_attr != nullptr) ? q_num_heads_attr->i() : 0;
  auto kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
  int64_t kv_num_heads = (kv_num_heads_attr != nullptr) ? kv_num_heads_attr->i() : 0;
  auto is_causal_attr = ctx.getAttribute("is_causal");
  int64_t is_causal = (is_causal_attr != nullptr) ? is_causal_attr->i() : 0;

  auto left_window_attr = ctx.getAttribute("left_window_size");
  int64_t left_window_size = (left_window_attr != nullptr) ? left_window_attr->i() : -1;
  auto right_window_attr = ctx.getAttribute("right_window_size");
  int64_t right_window_size = (right_window_attr != nullptr) ? right_window_attr->i() : -1;
  if (left_window_size < -1 || right_window_size < -1)
    return false;
  if (ctx.hasInput(4) != ctx.hasInput(5) || ctx.hasOutput(1) != ctx.hasOutput(2))
    return false;
  if (ctx.hasInput(6) && (ctx.hasInput(4) || ctx.hasOutput(1)))
    return false;
  if (t_qk->tensor_type().has_shape() && t_qk->tensor_type().shape().dim_size() == 4 &&
      (q_num_heads_attr != nullptr || kv_num_heads_attr != nullptr))
    return false;

  // Determine if input is 3D (requires reshape and transpose) or 4D (direct reshape)
  bool is_3d_input = (q_num_heads > 0 && kv_num_heads > 0);

  FunctionBuilder builder(functionProto);
  builder
      .Add("BatchSize = Shape <start = 0, end = 1> (Q)") // batch size
      .Add("QSeqLen = Shape <start = -2, end = -1> (Q)") // q_sequence_length
      .Add("KVSeqLen = Shape <start = -2, end = -1> (K)"); // kv_sequence_length

  if (is_3d_input) {
    // For 3D inputs: First reshape to [batch_size, seq_length, num_heads, head_size]
    // then transpose to [batch_size, num_heads, seq_length, head_size]
    builder
        .Const1D("QNumHeadsAttr", q_num_heads) // q_num_heads from attrs
        .Const1D("KVNumHeadsAttr", kv_num_heads) // kv_num_heads from attrs
        .Const1D("NegOne", static_cast<int64_t>(-1)); // head_size, inferred from other dimensions

    builder.Add("QIntermediateShape = Concat <axis = 0> (BatchSize, QSeqLen, QNumHeadsAttr, NegOne)")
        .Add("KVIntermediateShape = Concat <axis = 0> (BatchSize, KVSeqLen, KVNumHeadsAttr, NegOne)")
        .Add("QIntermediate = Reshape (Q, QIntermediateShape)")
        .Add("KIntermediate = Reshape (K, KVIntermediateShape)")
        .Add("VIntermediate = Reshape (V, KVIntermediateShape)")
        // Then transpose to [batch_size, num_heads, seq_length, head_size]
        .Add("QReshaped = Transpose <perm = [0, 2, 1, 3]> (QIntermediate)")
        .Add("KReshaped = Transpose <perm = [0, 2, 1, 3]> (KIntermediate)")
        .Add("VReshaped = Transpose <perm = [0, 2, 1, 3]> (VIntermediate)");
  } else {
    // For 4D inputs: Already in desired shape [batch_size, num_heads, seq_length, head_size]
    builder.Add("QReshaped = Identity(Q)").Add("KReshaped = Identity(K)").Add("VReshaped = Identity(V)");
  }

  builder
      .Add("QNumHeads = Shape <start = 1, end = 2> (QReshaped)") // q_num_heads
      .Add("KVNumHeads = Shape <start = 1, end = 2> (KReshaped)"); // kv_num_heads

  // Calculate scaling factor if scale attribute not provided
  auto scale_attr = ctx.getAttribute("scale");
  float scale = (scale_attr != nullptr) ? scale_attr->f() : static_cast<float>(1);
  builder
      .Add("QKHeadSize = Shape <start = 3, end = 4> (QReshaped)") // head_size for Q and K
      .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", float_type)
      .Add("VHeadSize = Shape <start = 3, end = 4> (VReshaped)") // head_size for V
      .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
      .Const1D("One1D", static_cast<int64_t>(1))
      .Const1D("NegOne1D", static_cast<int64_t>(-1))
      .Const1D("One1DF", static_cast<float>(1))
      .Const1D("Zero1D", static_cast<int64_t>(0))
      .Add("CalculatedScale = Div(One1DF, SqrtHeadSize)")
      .Const("ScaleF", ToTensor<float>(scale))
      .Add(scale_attr != nullptr ? "ScaleFactor = Identity(ScaleF)" : "ScaleFactor = Identity(CalculatedScale)")
      .Add("ScaleFactorSqrt = Sqrt(ScaleFactor)")
      .Add("ScaleFactorF = Cast (ScaleFactorSqrt)", "to", T1);

  // Update key and value caches for past and present states

  if (ctx.hasInput(4)) {
    builder.Add("PresentKey = Concat <axis = 2> (past_key, KReshaped)");
    builder.Add("PastKVSeqLen =  Shape <start = -2, end = -1> (past_key)");
  } else {
    builder.Add("PresentKey = Identity (KReshaped)");
    builder.Const1D("PastKVSeqLen", static_cast<int64_t>(0));
  }
  // External/static cache bottom-right offset (per batch): nonpad_kv_seqlen - q_len.
  // Only meaningful when is_causal=1, nonpad present (input 6), and no past_key (input 4).
  if (is_causal == 1 && ctx.hasInput(6) && !ctx.hasInput(4)) {
    builder.Add("CausalOffsetPerBatch = Sub(nonpad_kv_seqlen, QSeqLen)"); // (batch,)
  }
  if (ctx.hasOutput(1)) {
    builder.Add("present_key = Identity (PresentKey)");
  }

  if (ctx.hasInput(5)) {
    builder.Add("PresentValue = Concat <axis = 2> (past_value, VReshaped)");
  } else {
    builder.Add("PresentValue = Identity (VReshaped)");
  }
  if (ctx.hasOutput(2)) {
    builder.Add("present_value = Identity (PresentValue)");
  }

  if (!defs::nn::utils::AttentionAppendFunctionCausalMask(ctx, builder, true, true))
    return false;

  // Window bounds are an additive overlay on the causal and attention masks.
  if (left_window_size >= 0 || right_window_size >= 0) {
    builder.Const1D("WinZero", static_cast<int64_t>(0))
        .Const1D("WinOne", static_cast<int64_t>(1))
        .Add("WinZeroNoDim = Squeeze(WinZero, WinZero)")
        .Add("WinOneNoDim = Squeeze(WinOne, WinZero)")
        .Add("WinSeqLen = Squeeze(QSeqLen, WinZero)")
        .Add("WinTotalSeqLen = Squeeze(NewKVSeqLen, WinZero)")
        .Add("WinRangeRow = Range(WinZeroNoDim, WinSeqLen, WinOneNoDim)") // [Sq]
        .Add("WinRangeCol = Range(WinZeroNoDim, WinTotalSeqLen, WinOneNoDim)"); // [Skv]
    if (left_window_size >= 0) {
      builder.Const1D("LeftWindowSize", left_window_size).Add("LeftWindowSizeNoDim = Squeeze(LeftWindowSize, WinZero)");
    }
    if (right_window_size >= 0) {
      builder.Const1D("RightWindowSize", right_window_size)
          .Add("RightWindowSizeNoDim = Squeeze(RightWindowSize, WinZero)");
    }
    bool win_external_cache = ctx.hasInput(6) && !ctx.hasInput(4);
    if (win_external_cache) {
      // External cache: per-batch offset -> 4D mask
      // [batch, 1, Sq, Skv]
      builder
          .Add("WinOffset = Sub(nonpad_kv_seqlen, QSeqLen)") // (batch,)
          .Const("WinAxes123", std::vector<int64_t>{1, 2, 3})
          .Const("WinAxes01", std::vector<int64_t>{0, 1})
          .Add("WinOffset4D = Unsqueeze(WinOffset, WinAxes123)") // (batch,1,1,1)
          .Add("WinRow2D = Unsqueeze(WinRangeRow, WinOne)") // (Sq,1)
          .Add("WinRow4D = Unsqueeze(WinRow2D, WinAxes01)") // (1,1,Sq,1)
          .Add("WinCol2D = Unsqueeze(WinRangeCol, WinZero)") // (1,Skv)
          .Add("WinCol4D = Unsqueeze(WinCol2D, WinAxes01)") // (1,1,1,Skv)
          .Add("WinAbsPos = Add(WinRow4D, WinOffset4D)") // (batch,1,Sq,1)
          .Add("WinDiff = Sub(WinAbsPos, WinCol4D)"); // (batch,1,Sq,Skv)
    } else {
      // Internal cache (scalar PastKVSeqLen) or no cache: 2D mask [Sq, Skv]
      builder
          .Add("WinRow2D = Unsqueeze(WinRangeRow, WinOne)") // (Sq,1)
          .Add("WinCol2D = Unsqueeze(WinRangeCol, WinZero)") // (1,Skv)
          .Add("WinAbsPos = Add(WinRow2D, PastKVSeqLen)") // (Sq,1)
          .Add("WinDiff = Sub(WinAbsPos, WinCol2D)"); // (Sq,Skv)
    }
    if (left_window_size >= 0 && right_window_size >= 0) {
      builder.Add("WinLeftOk = LessOrEqual(WinDiff, LeftWindowSizeNoDim)")
          .Add("WinRightDiff = Neg(WinDiff)")
          .Add("WinRightOk = LessOrEqual(WinRightDiff, RightWindowSizeNoDim)")
          .Add("WinOk = And(WinLeftOk, WinRightOk)");
    } else if (left_window_size >= 0) {
      builder.Add("WinOk = LessOrEqual(WinDiff, LeftWindowSizeNoDim)");
    } else {
      builder.Add("WinRightDiff = Neg(WinDiff)").Add("WinOk = LessOrEqual(WinRightDiff, RightWindowSizeNoDim)");
    }
    builder.Add("WinMaskFloat = Where(WinOk, ScalarZero, FloatNegInf)")
        .Add("WinMask = CastLike(WinMaskFloat, AttnBiasCausalOrNot)")
        .Add("AttnBiasCausalWindow = Add(AttnBiasCausalOrNot, WinMask)");
  } else {
    builder.Add("AttnBiasCausalWindow = Identity(AttnBiasCausalOrNot)");
  }

  // Add padding mask if kv_nonpad_seqlen is provided
  if (ctx.hasInput(6)) {
    builder
        .Add("KVSeqLenExpanded = Unsqueeze(nonpad_kv_seqlen, One1D)") // [batch_size, 1]
        .Add("KVSeqLen0D = Squeeze(KVSeqLen)")
        .Const("Zero0D", static_cast<int64_t>(0))
        .Const("One0D", static_cast<int64_t>(1))
        .Add("Range = Range(Zero0D, KVSeqLen0D, One0D)") // [KVSeqLen,]
        .Add("PaddingMaskBool = Less(Range, KVSeqLenExpanded)") // [batch_size, KVSeqLen]
        .Add("PaddingMaskFloat = Where(PaddingMaskBool, ScalarZero, FloatNegInf)") // [batch_size, KVSeqLen]
        .Add("PaddingMask3D = Unsqueeze(PaddingMaskFloat, One1D)") // [batch_size, 1, KVSeqLen]
        .Add("PaddingMask4DFloat = Unsqueeze(PaddingMask3D, One1D)") // [batch_size, 1, 1, KVSeqLen]
        .Add("PaddingMask4D = CastLike(PaddingMask4DFloat, AttnBiasCausalWindow)")
        .Add("AttnBiasCausalPad = Add(AttnBiasCausalWindow, PaddingMask4D)");
  } else {
    builder.Add("AttnBiasCausalPad = Identity(AttnBiasCausalWindow)");
  }
  builder.Add("AttnBiasT = Cast (AttnBiasCausalPad)", "to", T1);

  // Group Query Attention is applied if the following are satisfied
  // 1) q_num_heads != kv_num_heads
  // 2) q_num_heads % kv_num_heads == 0
  // 3) kv_num_heads == k_num_heads == v_num_heads
  builder.Add("NGQACond1 = Equal(QNumHeads, KVNumHeads)")
      .Add("GQACond1 = Not(NGQACond1)")
      .Add("DivNumHeads = Div(QNumHeads, KVNumHeads)")
      .Add("IDivNumHeads = Cast(DivNumHeads)", "to", int_type)
      .Add("RemainderNumHeads = Mod(QNumHeads, KVNumHeads)")
      .Add("GQACond2 = Equal(RemainderNumHeads, Zero1D)")
      .Add("GQACond = And(GQACond1, GQACond2)")
      .Add("InterleaveDim = Where(GQACond, IDivNumHeads, One1D)");

  // repeat kv (repeat_interleave)
  builder.Const1D("Two1D", static_cast<int64_t>(2))
      .Add("KUnsqueezed = Unsqueeze(PresentKey, Two1D)") // [B, Hk, 1, T, Dk]
      .Add("VUnsqueezed = Unsqueeze(PresentValue, Two1D)"); // [B, Hk, 1, T, Dv]

  // Build expand shape: [B, Hk, repeats, T, Dk]
  builder.Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, QKHeadSize)")
      .Add("KExpanded = Expand(KUnsqueezed, KExpandShape)");
  builder.Add("VExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, VHeadSize)")
      .Add("VExpanded = Expand(VUnsqueezed, VExpandShape)");

  // Reshape to [B, Hq, T, Dk] where Hq = Hk * repeats
  builder.Add("KAttentionShape = Concat <axis = 0> (BatchSize, QNumHeads, NewKVSeqLen, QKHeadSize)")
      .Add("VAttentionShape = Concat <axis = 0> (BatchSize, QNumHeads, NewKVSeqLen, VHeadSize)")
      .Add("KAttentionInput = Reshape(KExpanded, KAttentionShape)")
      .Add("VAttentionInput = Reshape(VExpanded, VAttentionShape)");

  // The following pattern is applied
  //      Q          K          V
  //      |          |          |
  //     Q*scale    K*scale     |
  //      |          |          |
  //      |       Transpose     |
  //      |          |          |
  //      ---MatMul---          |
  //            |               |
  //  softcap (if provided)     |
  //            |               |
  // at_mask---Add              |
  //            |               |
  //         Softmax            |
  //            |               |
  //            -----MatMul------
  //                    |
  //                    Y
  builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KAttentionInput)")
      .Add("QScaled = Mul(QReshaped, ScaleFactorF)")
      .Add("KScaled = Mul(KTranspose, ScaleFactorF)")
      .Add("QKAttnWeight = MatMul(QScaled, KScaled)")
      .Add("QKAttnCast = Cast (QKAttnWeight)", "to", T1);

  // Apply softcap before bias/mask addition.
  // Softcap must come before mask so that -inf mask values remain -inf
  // (zero probability in softmax). If applied after, -inf maps to
  // -softcap (finite), leaking probability to masked positions.
  auto softcap_attr = ctx.getAttribute("softcap");
  float softcap_val = (softcap_attr != nullptr) ? softcap_attr->f() : static_cast<float>(0);
  if (softcap_val != 0) {
    builder.Const1D("Softcap", softcap_val)
        .Add("SoftcapF = Cast (Softcap)", "to", T1)
        .Add("SoftcapDiv = Div(QKAttnCast, SoftcapF)")
        .Add("SoftcapTanh = Tanh(SoftcapDiv)")
        .Add("QKAttnSoftcapped = Mul(SoftcapTanh, SoftcapF)")
        .Add("QKAttnWeightSoftcap = Add(QKAttnSoftcapped, AttnBiasT)");
  } else {
    builder.Add("QKAttnWeightSoftcap = Add(QKAttnCast, AttnBiasT)");
  }
  builder.Add("SoftmaxCast = Cast (QKAttnWeightSoftcap)", "to", softmax_precision)
      .Add("AttnWeightSoftmax = Softmax (SoftmaxCast)")
      .Add("SoftmaxOut = Cast (AttnWeightSoftmax)", "to", T1);

  // Fully-masked-row guard: a query row whose additive bias (AttnBiasT) is
  // entirely -inf (every key disallowed by the combined causal + attn_mask
  // constraints) softmaxes to NaN. Detect such rows on the additive bias
  // row-max and zero their probabilities with Where (not Mul; NaN * 0 = NaN)
  // BEFORE the P @ V contraction so 0 @ V = 0. The guard runs before the
  // mode-3 output capture so the exposed qk_matmul_output row is also zeroed,
  // consistent with Y. Mirrors the reference impl so primary == _expanded
  // bit-for-bit; a no-op for rows with any allowed key.
  builder.Add("BiasRowMaxAxes = Constant <value = int64[1] {-1}> ()")
      .Add("BiasRowMax = ReduceMax(AttnBiasT, BiasRowMaxAxes)") // keepdims=1 (default)
      .Add("FloatNegInfT = CastLike(FloatNegInf, AttnBiasT)")
      .Add("RowAllMasked = Equal(BiasRowMax, FloatNegInfT)")
      .Add("ZeroProbT = CastLike(ScalarZero, SoftmaxOut)")
      .Add("SoftmaxOutSafe = Where(RowAllMasked, ZeroProbT, SoftmaxOut)");

  // QK MatMul output if required
  auto qk_matmul_output_mode_attr = ctx.getAttribute("qk_matmul_output_mode");
  int64_t qk_matmul_output_mode = (qk_matmul_output_mode_attr != nullptr) ? qk_matmul_output_mode_attr->i() : 0;
  if (ctx.hasOutput(3)) {
    if (qk_matmul_output_mode == 1) {
      // Mode 1: after softcap (before bias addition)
      if (softcap_val != 0) {
        builder.Add("qk_matmul_output = Identity(QKAttnSoftcapped)");
      } else {
        // No softcap applied, same as raw QK
        builder.Add("qk_matmul_output = Identity(QKAttnCast)");
      }
    } else if (qk_matmul_output_mode == 2) {
      // Mode 2: QK + softcap + bias (after softcap + bias addition)
      builder.Add("qk_matmul_output = Identity(QKAttnWeightSoftcap)");
    } else if (qk_matmul_output_mode == 3) {
      // Mode 3: post-softmax, after the fully-masked-row guard (a fully-masked
      // row is zeroed, consistent with the primary output Y).
      builder.Add("qk_matmul_output = Identity(SoftmaxOutSafe)");
    } else {
      builder.Add("qk_matmul_output = Identity(QKAttnWeight)");
    }
  }

  builder.Add("YPreReshape = MatMul(SoftmaxOutSafe, VAttentionInput)");
  // Reshape Y to 3D if input is a 3D tensor
  if (is_3d_input) {
    builder.Add("YTranspose = Transpose <perm = [0, 2, 1, 3]> (YPreReshape)")
        .Add("YNewShape = Concat <axis = 0> (Zero1D, Zero1D, NegOne)")
        .Add("Y = Reshape(YTranspose, YNewShape)");
  } else {
    builder.Add("Y = Identity(YPreReshape)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    25,
    generated::ai_onnx::Attention_ver25()
        .TypeAndShapeInferenceFunction(Attention25Inference)
        .SetContextDependentFunctionBodyBuilder(AttentionFunctionBody_opset25));

static bool CausalConvWithStateFunctionBody_opset27(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // CausalConvWithState <activation> (input, weight, bias?, past_state?)
  //   => (output, present_state)
  //
  // Reference decomposition:
  //   1. Left-pad input with past_state (or zeros if absent) along axis 2
  //   2. Depthwise Conv1d (group=C, no padding) on the padded input
  //   3. Optional SiLU/Swish activation
  //   4. Slice last (k-1) positions from padded input as present_state

  // --- Step 1: Extract static channel count for Conv group attribute ---
  // Conv's group attribute must be a compile-time integer. Prefer the
  // input channel dimension when concrete; otherwise fall back to the
  // leading weight dimension, which for depthwise weights is C and is
  // typically statically known because weights are commonly
  // initializers.
  auto* input_type = ctx.getInputType(0);
  if (input_type == nullptr || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape() ||
      input_type->tensor_type().shape().dim_size() < 2) {
    return false;
  }
  int64_t C = 0;
  if (input_type->tensor_type().shape().dim(1).has_dim_value()) {
    C = input_type->tensor_type().shape().dim(1).dim_value();
  } else {
    auto* weight_type = ctx.getInputType(1);
    if (weight_type == nullptr || !weight_type->has_tensor_type() || !weight_type->tensor_type().has_shape() ||
        weight_type->tensor_type().shape().dim_size() < 1 ||
        !weight_type->tensor_type().shape().dim(0).has_dim_value()) {
      return false;
    }
    C = weight_type->tensor_type().shape().dim(0).dim_value();
  }

  // Read the activation attribute (default: "none")
  auto* activation_attr = ctx.getAttribute("activation");
  std::string activation = (activation_attr != nullptr && activation_attr->has_s()) ? activation_attr->s() : "none";

  FunctionBuilder builder(functionProto);

  // --- Shared kernel-size helpers (used by Step 2 zero-pad and Step 5 slice) ---
  builder.Const1D("One1D", static_cast<int64_t>(1)).Add(R"ONNX(
                KernelSize = Shape <start = 2, end = 3> (weight)
                Km1 = Sub (KernelSize, One1D)
              )ONNX");

  // --- Step 2: Build the left-padded input (B, C, L+k-1) ---
  if (ctx.hasInput(3)) {
    // past_state is provided: concat it on the left along axis 2
    builder.Add("PaddedInput = Concat <axis = 2> (past_state, input)");
  } else {
    // No past_state: zero-pad on the left by (k-1)
    builder.Add(R"ONNX(
                  BatchDim = Shape <start = 0, end = 1> (input)
                  ChannelDim = Shape <start = 1, end = 2> (input)
                  ZeroPadShape = Concat <axis = 0> (BatchDim, ChannelDim, Km1)
                  ZeroPad = ConstantOfShape (ZeroPadShape)
                  ZeroPadCast = CastLike (ZeroPad, input)
                  PaddedInput = Concat <axis = 2> (ZeroPadCast, input)
                )ONNX");
  }

  // --- Step 3: Depthwise Conv1d (group=C, valid padding) ---
  if (ctx.hasInput(2)) {
    // Bias provided: pass directly to Conv
    builder.Add("ConvOut = Conv (PaddedInput, weight, bias)", "group", C);
  } else {
    // No bias
    builder.Add("ConvOut = Conv (PaddedInput, weight)", "group", C);
  }

  // --- Step 4: Optional fused activation ---
  if (activation == "silu" || activation == "swish") {
    // Upcast Sigmoid/Mul to float32 to match the Python reference
    // implementation, which evaluates SiLU in float32 for fp16/bf16
    // inputs and casts back. For float32 inputs the Cast/CastLike
    // pair is a no-op that runtimes fold away.
    builder.Add(R"ONNX(
                  ConvOutFloat = Cast <to = 1> (ConvOut)
                  ConvSigmoid = Sigmoid (ConvOutFloat)
                  MulOutFloat = Mul (ConvOutFloat, ConvSigmoid)
                  output = CastLike (MulOutFloat, ConvOut)
                )ONNX");
  } else if (activation == "none") {
    builder.Add("output = Identity (ConvOut)");
  } else {
    // Unsupported activation values are rejected up front by
    // TypeAndShapeInferenceFunction; if we somehow reach the body
    // builder with an unknown value (e.g. inference was bypassed),
    // decline to build a function body rather than throw.
    return false;
  }

  // --- Step 5: Slice last (k-1) positions from PaddedInput as present_state ---
  // present_state = PaddedInput[:, :, L : L+(k-1)]
  // Compute start/end from PaddedInput's actual length so the k=1 (Km1=0)
  // edge case yields an empty (B, C, 0) slice. A negative start index
  // would not work here because Slice treats -0 as 0, returning the full
  // padded sequence instead of an empty tensor.
  builder
      .Add(R"ONNX(
                    PadLen = Shape <start = 2, end = 3> (PaddedInput)
                    StartIdx = Sub (PadLen, Km1)
                  )ONNX")
      .Const("SliceAxes", std::vector<int64_t>{2})
      .Add("present_state = Slice (PaddedInput, StartIdx, PadLen, SliceAxes)");

  schema.BuildFunction(functionProto);
  return true;
}

static void CausalConvWithStateShapeInference_opset27(InferenceContext& ctx) {
  // --- Validate activation attribute ---
  // Reject unknown values here (rather than in the function-body builder)
  // so checker/shape-inference users get a clear error even when the op
  // is never inlined. Allowed values mirror the schema doc string and
  // the Python reference implementation.
  auto* activation_attr = ctx.getAttribute("activation");
  if (activation_attr != nullptr && activation_attr->has_s()) {
    const std::string& activation = activation_attr->s();
    if (activation != "none" && activation != "silu" && activation != "swish") {
      fail_shape_inference(
          "CausalConvWithState: unsupported activation value '",
          activation,
          "'. Supported values are 'none', 'silu', and 'swish'.");
    }
  }

  // --- Type inference ---
  // All inputs and outputs share the same type T.
  propagateElemTypeFromInputToOutput(ctx, 0, 0); // input -> output
  propagateElemTypeFromInputToOutput(ctx, 0, 1); // input -> present_state

  // --- Output 0: same shape as input (B, C, L) ---
  propagateShapeFromInputToOutput(ctx, 0, 0);

  // --- Shape inference via dimension unification ---
  // Declare logical dims and unify each input against its expected shape.
  // unifyInputShape implicitly validates the rank of each provided input
  // and propagates the most-informative known value across inputs that
  // share a dim (e.g. B and C across input/weight/bias/past_state, and
  // K between weight and past_state).
  Dim B, C, L, One, K, KMinus1;
  One.set_dim_value(1);
  ctx.unifyInputShape(0, {B, C, L});
  ctx.unifyInputShape(1, {C, One, K});
  if (ctx.hasInput(2)) {
    ctx.unifyInputShape(2, {C});
  }
  if (ctx.hasInput(3)) {
    ctx.unifyInputShape(3, {B, C, KMinus1});
  }

  // Connect KMinus1 to K (when K is known) using Dim arithmetic.
  // If both are known via past_state and weight, unifyDim checks they agree.
  unifyDim(K - 1, KMinus1);

  if (K.has_dim_value() && K.dim_value() < 1) {
    fail_shape_inference("Kernel size k must be >= 1, got ", K.dim_value());
  }

  // --- Output 1: present_state shape (B, C, k - 1) ---
  updateOutputShape(ctx, 1, {B, C, KMinus1});
}

ONNX_OPERATOR_SET_SCHEMA(
    CausalConvWithState,
    27,
    generated::ai_onnx::CausalConvWithState_ver27()
        .TypeAndShapeInferenceFunction(CausalConvWithStateShapeInference_opset27)
        .SetContextDependentFunctionBodyBuilder(CausalConvWithStateFunctionBody_opset27));

static void LinearAttentionShapeInference_opset27(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Read required attributes
  auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
  auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
  int64_t q_num_heads = (q_num_heads_attr && q_num_heads_attr->has_i()) ? q_num_heads_attr->i() : 0;
  int64_t kv_num_heads = (kv_num_heads_attr && kv_num_heads_attr->has_i()) ? kv_num_heads_attr->i() : 0;

  // GQA constraint: q_num_heads must be a positive multiple of kv_num_heads.
  // q_num_heads == kv_num_heads is MHA; q_num_heads > kv_num_heads with the
  // divisibility rule is GQA (or MQA when kv_num_heads == 1). Non-positive
  // head counts are rejected here so that downstream divisions and the
  // function body always see strictly positive values.
  if (q_num_heads <= 0) {
    fail_type_inference("q_num_heads must be > 0, got ", q_num_heads);
  }
  if (kv_num_heads <= 0) {
    fail_type_inference("kv_num_heads must be > 0, got ", kv_num_heads);
  }
  if (q_num_heads % kv_num_heads != 0) {
    fail_type_inference("q_num_heads (", q_num_heads, ") must be a multiple of kv_num_heads (", kv_num_heads, ")");
  }

  // Read update_rule attribute with default
  auto* update_rule_attr = ctx.getAttribute("update_rule");
  std::string update_rule = (update_rule_attr != nullptr) ? update_rule_attr->s() : "gated_delta";

  // Validate update_rule value
  if (update_rule != "linear" && update_rule != "gated" && update_rule != "delta" && update_rule != "gated_delta") {
    fail_type_inference("update_rule must be one of: 'linear', 'gated', 'delta', 'gated_delta'");
  }

  // Validate update_rule vs optional inputs (decay=input 4, beta=input 5).
  const bool has_past_state = ctx.hasInput(3);
  const bool has_decay = ctx.hasInput(4);
  const bool has_beta = ctx.hasInput(5);

  if (update_rule == "linear") {
    if (has_decay) {
      fail_type_inference("update_rule 'linear' forbids decay input");
    }
    if (has_beta) {
      fail_type_inference("update_rule 'linear' forbids beta input");
    }
  } else if (update_rule == "gated") {
    if (!has_decay) {
      fail_type_inference("update_rule 'gated' requires decay input");
    }
    if (has_beta) {
      fail_type_inference("update_rule 'gated' forbids beta input");
    }
  } else if (update_rule == "delta") {
    if (has_decay) {
      fail_type_inference("update_rule 'delta' forbids decay input");
    }
    if (!has_beta) {
      fail_type_inference("update_rule 'delta' requires beta input");
    }
  } else if (update_rule == "gated_delta") {
    if (!has_decay) {
      fail_type_inference("update_rule 'gated_delta' requires decay input");
    }
    if (!has_beta) {
      fail_type_inference("update_rule 'gated_delta' requires beta input");
    }
  }

  // --- Type propagation ---
  // output: type T from query.
  //
  // present_state uses the separately-declared state type S only when S can be
  // anchored by the optional past_state input. When past_state is omitted,
  // type inference cannot independently infer S, so present_state explicitly
  // falls back to type T from query on the first step. Callers that require an
  // fp32 state while using lower-precision query activations must therefore
  // provide past_state with the desired state element type.
  // (The spec recommends S == float32 or S == T for numerical stability.)
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (has_past_state && ctx.getInputType(3) != nullptr && ctx.getInputType(3)->has_tensor_type()) {
    propagateElemTypeFromInputToOutput(ctx, 3, 1);
  } else {
    // No past_state is available to anchor S, so present_state defaults to T.
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
  }

  // --- Shape inference via dimension unification ---
  // Declare logical dims and unify each input against its expected shape.
  // unifyInputShape validates the rank of each provided input and merges
  // the most-informative known value across inputs that share a dim
  // (B and T across Q/K/V/decay/beta/past_state; H_kv between past_state
  // and the kv_num_heads attribute; d_k between past_state, Q (via
  // QPack/q_num_heads) and K; d_v between past_state and V).
  Dim B, T, QPack, KPack, VPack, Hkv, Dk, Dv;
  if (kv_num_heads > 0) {
    Hkv.set_dim_value(kv_num_heads);
  }
  ctx.unifyInputShape(0, {B, T, QPack});
  ctx.unifyInputShape(1, {B, T, KPack});
  ctx.unifyInputShape(2, {B, T, VPack});
  if (has_past_state) {
    ctx.unifyInputShape(3, {B, Hkv, Dk, Dv});
  }
  // decay shape: (B, T, H_kv * d_k) for per-key-dim or (B, T, H_kv) for
  // per-head. Both are rank-3; we only constrain B and T.
  if (has_decay) {
    Dim DecayLast;
    ctx.unifyInputShape(4, {B, T, DecayLast});
  }
  // beta shape: (B, T, H_kv) or (B, T, 1). Both rank-3; only B, T checked.
  if (has_beta) {
    Dim BetaLast;
    ctx.unifyInputShape(5, {B, T, BetaLast});
  }

  // Derive d_k from Q (via q_num_heads) and K (via kv_num_heads).
  // Multiple sources for the same logical dim are unified, so an
  // inconsistency between Q and K (e.g. mismatched d_k) fails here.
  // Reject packed last dims that are not divisible by their head count
  // up-front so the resulting integer-division d_k actually corresponds
  // to a shape the function body can reshape to (B, T, H, D).
  if (q_num_heads > 0 && QPack.has_dim_value()) {
    if (QPack.dim_value() % q_num_heads != 0) {
      fail_shape_inference(
          "Packed query last dim (", QPack.dim_value(), ") must be divisible by q_num_heads (", q_num_heads, ")");
    }
    unifyDim(Dk, checkedDivide(QPack.dim_value(), q_num_heads));
  }
  if (kv_num_heads > 0 && KPack.has_dim_value()) {
    if (KPack.dim_value() % kv_num_heads != 0) {
      fail_shape_inference(
          "Packed key last dim (", KPack.dim_value(), ") must be divisible by kv_num_heads (", kv_num_heads, ")");
    }
    unifyDim(Dk, checkedDivide(KPack.dim_value(), kv_num_heads));
  }
  // Derive d_v from V.
  if (kv_num_heads > 0 && VPack.has_dim_value()) {
    if (VPack.dim_value() % kv_num_heads != 0) {
      fail_shape_inference(
          "Packed value last dim (", VPack.dim_value(), ") must be divisible by kv_num_heads (", kv_num_heads, ")");
    }
    unifyDim(Dv, checkedDivide(VPack.dim_value(), kv_num_heads));
  }

  // Strict validation of decay/beta last dims, deferred until d_k and H_kv
  // are derived above. The earlier unifyInputShape calls only checked rank
  // and B/T; here we reject concrete last dims that the function body and
  // Python reference cannot broadcast correctly.
  //   decay last dim must be H_kv (per-head scalar form) or H_kv * d_k
  //     (per-key-dim form). When d_k is unknown we still require divisibility
  //     by H_kv.
  //   beta  last dim must be 1 (broadcast) or H_kv (per-head).
  if (has_decay && kv_num_heads > 0) {
    auto* decay_type = ctx.getInputType(4);
    if (decay_type != nullptr && decay_type->has_tensor_type() && decay_type->tensor_type().has_shape() &&
        decay_type->tensor_type().shape().dim_size() == 3 && decay_type->tensor_type().shape().dim(2).has_dim_value()) {
      const int64_t decay_last = decay_type->tensor_type().shape().dim(2).dim_value();
      if (Dk.has_dim_value()) {
        const int64_t dk = Dk.dim_value();
        const int64_t kv_dk = checkedMultiply(kv_num_heads, dk);
        if (decay_last != kv_num_heads && decay_last != kv_dk) {
          fail_shape_inference(
              "decay last dim (",
              decay_last,
              ") must be kv_num_heads (",
              kv_num_heads,
              ") or kv_num_heads * d_k (",
              kv_dk,
              ")");
        }
      } else if (decay_last != kv_num_heads && decay_last % kv_num_heads != 0) {
        fail_shape_inference(
            "decay last dim (",
            decay_last,
            ") must be kv_num_heads (",
            kv_num_heads,
            ") or a multiple of kv_num_heads");
      }
    }
  }
  if (has_beta && kv_num_heads > 0) {
    auto* beta_type = ctx.getInputType(5);
    if (beta_type != nullptr && beta_type->has_tensor_type() && beta_type->tensor_type().has_shape() &&
        beta_type->tensor_type().shape().dim_size() == 3 && beta_type->tensor_type().shape().dim(2).has_dim_value()) {
      const int64_t beta_last = beta_type->tensor_type().shape().dim(2).dim_value();
      if (beta_last != 1 && beta_last != kv_num_heads) {
        fail_shape_inference("beta last dim (", beta_last, ") must be 1 or kv_num_heads (", kv_num_heads, ")");
      }
    }
  }

  // Output 0: (B, T, H_q * d_v) — 3D packed.
  Dim OutLast;
  if (q_num_heads > 0 && Dv.has_dim_value()) {
    OutLast.set_dim_value(checkedMultiply(q_num_heads, Dv.dim_value()));
  }
  updateOutputShape(ctx, 0, {B, T, OutLast});

  // Output 1: present_state (B, H_kv, d_k, d_v) — always 4D.
  updateOutputShape(ctx, 1, {B, Hkv, Dk, Dv});
}

static bool LinearAttentionFunctionBody_opset27(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // LinearAttention <update_rule, scale, q_num_heads, kv_num_heads, chunk_size>
  //   (query, key, value, past_state?, decay?, beta?)
  //   => (output, present_state)
  //
  // Reference decomposition (sequential recurrence via Scan):
  //   1. Reshape/transpose 3D packed inputs (B, T, H*D) -> 4D (B, H, T, D)
  //   2. Initialize state (past_state or zeros), shape (B, H_kv, d_k, d_v)
  //   3. Compute scale factor (attribute or 1/sqrt(d_k))
  //   4. Run Scan over T axis with body implementing one recurrence step
  //   5. Transpose/reshape 4D outputs back to 3D (B, T, H_q*d_v)

  // --- Step 1: Extract attributes and validate ---
  auto* update_rule_attr = ctx.getAttribute("update_rule");
  std::string update_rule =
      (update_rule_attr != nullptr && update_rule_attr->has_s()) ? update_rule_attr->s() : "gated_delta";

  auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
  auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
  if (q_num_heads_attr == nullptr || !q_num_heads_attr->has_i() || kv_num_heads_attr == nullptr ||
      !kv_num_heads_attr->has_i()) {
    return false;
  }
  int64_t q_num_heads = q_num_heads_attr->i();
  int64_t kv_num_heads = kv_num_heads_attr->i();

  auto* scale_attr = ctx.getAttribute("scale");
  float scale = (scale_attr != nullptr && scale_attr->has_f()) ? scale_attr->f() : 0.0f;

  // Per-rule presence of optional inputs
  bool has_decay = (update_rule == "gated" || update_rule == "gated_delta");
  bool has_beta = (update_rule == "delta" || update_rule == "gated_delta");

  // Need query type (for CastLike on zero state) — checked indirectly via input 0 presence.
  if (ctx.getInputType(0) == nullptr) {
    return false;
  }

  // GQA: q_num_heads must be a positive multiple of kv_num_heads.
  // group_size == 1 is the standard MHA case (no expansion); group_size > 1
  // means each KV head is shared by `group_size` query heads.
  if (q_num_heads <= 0 || kv_num_heads <= 0 || q_num_heads % kv_num_heads != 0) {
    return false;
  }
  int64_t group_size = q_num_heads / kv_num_heads;

  FunctionBuilder builder(functionProto);

  // --- Step 2: Pre-Scan 3D -> 4D reshape and transpose ---
  // Inputs arrive packed as (B, T, H*D). Reshape to (B, T, H, D) then
  // transpose to (B, H, T, D) so the Scan iterates along the T axis.
  builder.Const1D("NegOne1D", static_cast<int64_t>(-1))
      .Const1D("One1D", static_cast<int64_t>(1))
      .Const1D("HqConst", q_num_heads)
      .Const1D("HkvConst", kv_num_heads)
      .Add(R"ONNX(
                BatchDim = Shape <start = 0, end = 1> (query)
                SeqLenDim = Shape <start = 1, end = 2> (query)
              )ONNX");

  // Inputs are reshaped to (B, T, H, D), then transposed so the T axis is
  // first: (T, B, H, D). The leading T axis is consumed by Scan (which
  // only supports scan_input_axes=0 in the ONNX reference runtime).

  // Query: (B, T, H_q*d_k) -> (B, T, H_q, d_k) -> (T, B, H_q, d_k)
  builder.Add(R"ONNX(
                QShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, HqConst, NegOne1D)
                QReshaped = Reshape (query, QShape4D)
                Q4D = Transpose <perm = [1, 0, 2, 3]> (QReshaped)
              )ONNX");

  // Key/Value/Decay all share H_kv as the head dimension; -1 absorbs
  // d_k (key/decay) or d_v (value), and degenerates to 1 for per-head
  // scalar decay of shape (B, T, H_kv).
  builder.Add(R"ONNX(
                KVShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, HkvConst, NegOne1D)
                KReshaped = Reshape (key, KVShape4D)
                K4D = Transpose <perm = [1, 0, 2, 3]> (KReshaped)
                VReshaped = Reshape (value, KVShape4D)
                V4D = Transpose <perm = [1, 0, 2, 3]> (VReshaped)
              )ONNX");

  if (has_decay) {
    // Decay: (B, T, H_kv*d_k) -> (T, B, H_kv, d_k), or
    //        (B, T, H_kv)      -> (T, B, H_kv, 1)   [broadcastable].
    builder.Add(R"ONNX(
                  DecayReshaped = Reshape (decay, KVShape4D)
                  Decay4D = Transpose <perm = [1, 0, 2, 3]> (DecayReshaped)
                )ONNX");
  }

  if (has_beta) {
    // Beta: (B, T, H_kv) or (B, T, 1) -> reshape (B, T, -1, 1)
    //       -> transpose (T, B, H_kv_or_1, 1) [broadcastable head dim].
    builder.Add(R"ONNX(
                  BetaShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, NegOne1D, One1D)
                  BetaReshaped = Reshape (beta, BetaShape4D)
                  Beta4D = Transpose <perm = [1, 0, 2, 3]> (BetaReshaped)
                )ONNX");
  }

  // --- Step 3: Initialize recurrence state (B, H_kv, d_k, d_v) ---
  // Compute d_k and d_v as 1D shape tensors for downstream use
  // (state shape here, scale factor in Step 4).
  builder.Add(R"ONNX(
                QLastDim = Shape <start = 2, end = 3> (query)
                VLastDim = Shape <start = 2, end = 3> (value)
                DkDim = Div (QLastDim, HqConst)
                DvDim = Div (VLastDim, HkvConst)
              )ONNX");

  // State and scale are kept in float32 inside the body to match the
  // Python reference's fp32 accumulation. `present_state` is cast back
  // to S (past_state's dtype, or query's dtype when past_state is
  // absent) after the Scan; `output` is cast back to T inside the body.
  if (ctx.hasInput(3)) {
    // past_state has shape (B, H_kv, d_k, d_v) and dtype S.
    // Upcast to float32 for accumulation regardless of S.
    builder.Add("State0 = Cast <to = 1> (past_state)");
  } else {
    // ConstantOfShape with default value yields float32 already; no
    // cast required. S anchor for the post-Scan cast defaults to T
    // (query's dtype) in this case.
    builder.Add(R"ONNX(
                  StateShape = Concat <axis = 0> (BatchDim, HkvConst, DkDim, DvDim)
                  State0 = ConstantOfShape (StateShape)
                )ONNX");
  }

  // --- Step 4: Compute scale factor (scalar, float32) ---
  // If `scale` attribute is non-zero, use it directly; otherwise
  // derive 1/sqrt(d_k) from query's last dim. ScaleFactor stays in
  // float32 to multiply against the float32 read result inside the
  // body; the body casts the final output back to T.
  if (scale != 0.0f) {
    builder.Const("ScaleFactor", scale);
  } else {
    builder.Add(R"ONNX(
                  DkScalar = Squeeze (DkDim)
                  DkFloat = Cast <to = 1> (DkScalar)
                  SqrtDk = Sqrt (DkFloat)
                  ScaleFactor = Reciprocal (SqrtDk)
                )ONNX");
  }

  // --- Step 4b: Precompute GQA expand/read shapes (only when group_size > 1) ---
  // GQA expansion uses the same Unsqueeze + Expand + Reshape pattern as the
  // standard Attention op. The state lives at H_kv heads (one state per
  // KV head — recurrence is per-KV-head); only the read step needs to
  // broadcast it across the `group_size` query heads sharing that KV head.
  // Shapes are computed once outside the Scan body and captured inside
  // by name.
  if (group_size > 1) {
    builder.Const1D("GroupSize", group_size).Const1D("Two1D", static_cast<int64_t>(2)).Add(R"ONNX(
                  StateExpandShape = Concat <axis = 0> (BatchDim, HkvConst, GroupSize, DkDim, DvDim)
                  StateReadShape = Concat <axis = 0> (BatchDim, HqConst, DkDim, DvDim)
                )ONNX");
  }

  // --- Step 5: Scan node with body subgraph ---
  // The body implements one recurrence step. We write one complete
  // body per update_rule (linear / gated / delta / gated_delta) so
  // the recurrence is readable inline rather than assembled from
  // fragments. ScaleFactor (and, for group_size > 1,
  // StateExpandShape / StateReadShape) are referenced from the
  // enclosing scope (Scan subgraphs may capture outer-scope
  // tensors by name).
  //
  // The body always accumulates in float32 to match the Python
  // reference: each sliced T-typed input (q_t/k_t/v_t/decay_t/beta_t)
  // is cast to float32 at body entry; state_in/state_out stay
  // in float32 throughout; output_t is cast back to T (anchored on
  // q_t) before being emitted as a per-step Scan output. The post-
  // Scan code casts FinalState back to S.
  //
  // Axes constants used inside the body for Unsqueeze/Squeeze:
  // NegOne1D was added in Step 2; Two1D is set up in Step 4b
  // when group_size > 1.
  builder.Const1D("NegTwo1D", static_cast<int64_t>(-2));

  // Read block (shared across all update_rules; differs only by GQA):
  // output_t = ScaleFactor * (q_t^T @ state_out), then CastLike to T.
  // For GQA (group_size > 1) the per-KV-head state must first be
  // repeat-interleaved across the `group_size` query heads sharing
  // each KV head, using the same Unsqueeze + Expand + Reshape
  // pattern as the standard Attention op. `StateExpandShape` and
  // `StateReadShape` are captured from the enclosing scope.
  const std::string read_block = (group_size > 1) ? R"ONNX(
                StateUnsq = Unsqueeze (state_out, Two1D)
                StateExpanded = Expand (StateUnsq, StateExpandShape)
                StateForRead = Reshape (StateExpanded, StateReadShape)
                QtRow = Unsqueeze (q_t_f, NegTwo1D)
                ReadOut = MatMul (QtRow, StateForRead)
                ReadSq = Squeeze (ReadOut, NegTwo1D)
                output_t_f = Mul (ScaleFactor, ReadSq)
                output_t = CastLike (output_t_f, q_t)
              )ONNX"
                                                  : R"ONNX(
                QtRow = Unsqueeze (q_t_f, NegTwo1D)
                ReadOut = MatMul (QtRow, state_out)
                ReadSq = Squeeze (ReadOut, NegTwo1D)
                output_t_f = Mul (ScaleFactor, ReadSq)
                output_t = CastLike (output_t_f, q_t)
              )ONNX";

  // One complete recurrence body per update_rule. Each body:
  //   * casts its T-typed inputs to float32,
  //   * computes state_out = (decayed) state_in + KtCol @ WriteRow,
  //     where WriteRow comes from v_t_f directly (linear/gated) or
  //     from a delta-rule correction Delta = beta * (v - retrieved).
  std::string body;
  std::string body_inputs;
  std::string scan_inputs;
  int64_t num_scan_inputs;
  if (update_rule == "linear") {
    body_inputs = "state_in, q_t, k_t, v_t";
    scan_inputs = "State0, Q4D, K4D, V4D";
    num_scan_inputs = 3;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                WriteRow = Unsqueeze (v_t_f, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (state_in, Outer)
              )ONNX";
  } else if (update_rule == "gated") {
    body_inputs = "state_in, q_t, k_t, v_t, decay_t";
    scan_inputs = "State0, Q4D, K4D, V4D, Decay4D";
    num_scan_inputs = 4;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                decay_t_f = Cast <to = 1> (decay_t)
                DecayExp = Exp (decay_t_f)
                DecayExpUnsq = Unsqueeze (DecayExp, NegOne1D)
                DecayedState = Mul (state_in, DecayExpUnsq)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                WriteRow = Unsqueeze (v_t_f, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (DecayedState, Outer)
              )ONNX";
  } else if (update_rule == "delta") {
    body_inputs = "state_in, q_t, k_t, v_t, beta_t";
    scan_inputs = "State0, Q4D, K4D, V4D, Beta4D";
    num_scan_inputs = 4;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                beta_t_f = Cast <to = 1> (beta_t)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                StateT = Transpose <perm = [0, 1, 3, 2]> (state_in)
                Retrieved = MatMul (StateT, KtCol)
                RetrievedSq = Squeeze (Retrieved, NegOne1D)
                Diff = Sub (v_t_f, RetrievedSq)
                Delta = Mul (beta_t_f, Diff)
                WriteRow = Unsqueeze (Delta, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (state_in, Outer)
              )ONNX";
  } else { // gated_delta
    body_inputs = "state_in, q_t, k_t, v_t, decay_t, beta_t";
    scan_inputs = "State0, Q4D, K4D, V4D, Decay4D, Beta4D";
    num_scan_inputs = 5;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                decay_t_f = Cast <to = 1> (decay_t)
                beta_t_f = Cast <to = 1> (beta_t)
                DecayExp = Exp (decay_t_f)
                DecayExpUnsq = Unsqueeze (DecayExp, NegOne1D)
                DecayedState = Mul (state_in, DecayExpUnsq)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                StateT = Transpose <perm = [0, 1, 3, 2]> (DecayedState)
                Retrieved = MatMul (StateT, KtCol)
                RetrievedSq = Squeeze (Retrieved, NegOne1D)
                Diff = Sub (v_t_f, RetrievedSq)
                Delta = Mul (beta_t_f, Diff)
                WriteRow = Unsqueeze (Delta, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (DecayedState, Outer)
              )ONNX";
  }

  // Assemble Scan node text. Body name is arbitrary.
  // We omit scan_input_axes/scan_output_axes (default 0); Q/K/V/decay/beta
  // were already pre-transposed so the T axis is the leading dimension.
  std::string scan_text =
      "FinalState, OutputAccum = Scan <\n"
      "  num_scan_inputs = " +
      std::to_string(num_scan_inputs) +
      ",\n"
      "  body = linear_attn_step (" +
      body_inputs + ") => (state_out, output_t) {\n" + body + read_block +
      "              }\n"
      "> (" +
      scan_inputs + ")";

  builder.Add(scan_text.c_str());

  // --- Step 6: Post-Scan 4D -> 3D reshape and present_state ---
  // OutputAccum: (T, B, H_q, d_v) -> Transpose to (B, T, H_q, d_v)
  // -> Reshape to (B, T, H_q * d_v). Already T-typed (cast inside body).
  // present_state: cast FinalState (float32) back to S.
  //   * With past_state: anchor on past_state's dtype.
  //   * Without past_state: default S to T (query's dtype).
  builder.Add(R"ONNX(
                OutputTransposed = Transpose <perm = [1, 0, 2, 3]> (OutputAccum)
                OutputShape3D = Concat <axis = 0> (BatchDim, SeqLenDim, NegOne1D)
                output = Reshape (OutputTransposed, OutputShape3D)
              )ONNX");
  if (ctx.hasInput(3)) {
    builder.Add("present_state = CastLike (FinalState, past_state)");
  } else {
    builder.Add("present_state = CastLike (FinalState, query)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    LinearAttention,
    27,
    generated::ai_onnx::LinearAttention_ver27()
        .SetContextDependentFunctionBodyBuilder(LinearAttentionFunctionBody_opset27)
        .TypeAndShapeInferenceFunction(LinearAttentionShapeInference_opset27));
} // namespace ONNX_NAMESPACE
