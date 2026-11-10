// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

// For GlobalPool operations.
static void globalPoolTypeShapeInference_opset2(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // (N, C, 1, 1, ..., 1)
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

static std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator_opset2() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { globalPoolTypeShapeInference_opset2(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    2,
    generated::ai_onnx::GlobalLpPool_ver2().FillUsing(GlobalLpPoolingOpSchemaGenerator_opset2()));

static void convPoolShapeInference_opset19(
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
      auto input_dims_size = n_input_dims;
      for (size_t i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape.dim(static_cast<int>(2 + i)).has_dim_value()) {
            continue;
          }
          residual = input_shape.dim(static_cast<int>(2 + i)).dim_value();
          if (residual < 0) {
            continue;
          }
          residual %= stride;
        }
        if (i >= effective_kernel_shape.size()) {
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
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size = checkedAdd(effective_input_size, pads[i]);
    effective_input_size = checkedAdd(effective_input_size, pads[i + kernel_shape_size]);

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions = 0;

    const int64_t stride_gap = checkedSubtract(effective_input_size, effective_kernel_shape[i]);
    if (ceil_mode == 1)
      // exact ceil division; (a + b - 1) / b would be wrong for negative a
      strided_kernel_positions =
          checkedAdd(checkedDivide(stride_gap, strides[i]), (stride_gap % strides[i] > 0 ? 1 : 0));
    else
      strided_kernel_positions = checkedDivide(stride_gap, strides[i]);

    // add in the initial position
    newdim->set_dim_value(checkedAdd(1, strided_kernel_positions));
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    13,
    generated::ai_onnx::Dropout_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    1,
    generated::ai_onnx::LpNormalization_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    6,
    generated::ai_onnx::InstanceNormalization_ver6().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

static void maxUnpoolShapeInference_opset11(InferenceContext& ctx) {
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

// For GlobalPool operations.
static void globalPoolTypeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // (N, C, 1, 1, ..., 1)
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

static std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator_opset1() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { globalPoolTypeShapeInference_opset1(ctx); });
  };
}
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    1,
    generated::ai_onnx::GlobalAveragePool_ver1().FillUsing(GlobalPoolingOpSchemaGenerator_opset1()));
ONNX_OPERATOR_SET_SCHEMA(
    GlobalMaxPool,
    1,
    generated::ai_onnx::GlobalMaxPool_ver1().FillUsing(GlobalPoolingOpSchemaGenerator_opset1()));

static void convTransposeShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);
  if (group <= 0) {
    fail_shape_inference("Attribute group must be > 0 for ConvTranspose. group=", group, ".");
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 3) {
    fail_shape_inference(
        "Input tensor must have at least 3 dimensions (N x C x D1...Dn). Got: ", input_shape.dim_size());
  }
  const auto& input_channels_dim = input_shape.dim(1);
  if (input_channels_dim.has_dim_value() && input_channels_dim.dim_value() % group != 0) {
    fail_shape_inference(
        "Input channels C must be divisible by group for ConvTranspose. C=",
        input_channels_dim.dim_value(),
        " group=",
        group,
        ".");
  }

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

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    19,
    generated::ai_onnx::DeformConv_ver19().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset19(ctx, true, false, 0, 1);
    }));

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator_opset11() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { convTransposeShapeInference_opset11(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    11,
    generated::ai_onnx::ConvTranspose_ver11().FillUsing(ConvTransposeOpSchemaGenerator_opset11()));

static std::function<void(OpSchema&)> ConvOpSchemaGenerator_opset11() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset19(ctx, true, false, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Conv, 11, generated::ai_onnx::Conv_ver11().FillUsing(ConvOpSchemaGenerator_opset11()));

static void roiPoolTypeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // rois is the second input.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto rios_shape = ctx.getInputType(1)->tensor_type().shape();

  if (input_shape.dim_size() < 4) {
    fail_shape_inference("Input tensor must have at least 4 dimensions");
  }
  if (rios_shape.dim_size() != 2) {
    fail_shape_inference("RoIs tensor must have 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pooled_shape;
  if (getRepeatedAttribute(ctx, "pooled_shape", pooled_shape)) {
    if (pooled_shape.size() != n_input_dims) {
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

  // (num_rois, channels, pooled_shape[0], pooled_shape[1])
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *output_shape->add_dim() = rios_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);
  output_shape->add_dim()->set_dim_value(pooled_shape[0]);
  output_shape->add_dim()->set_dim_value(pooled_shape[1]);
}

static std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator_opset1() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { roiPoolTypeShapeInference_opset1(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    MaxRoiPool,
    1,
    generated::ai_onnx::MaxRoiPool_ver1().FillUsing(RoiPoolOpSchemaGenerator_opset1()));

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator_opset18() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset19(ctx, true, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 18, generated::ai_onnx::LpPool_ver18().FillUsing(LpPoolOpSchemaGenerator_opset18()));

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    11,
    generated::ai_onnx::MaxUnpool_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      maxUnpoolShapeInference_opset11(ctx);
    }));

static void PoolShapeInference_opset19(InferenceContext& ctx, bool use_dilation) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* output_type = ctx.getOutputType(1);
    if (output_type->value_case() == TypeProto::kTensorType || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
      output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    }
  }
  convPoolShapeInference_opset19(ctx, use_dilation, true, 0, 1);
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    19,
    generated::ai_onnx::AveragePool_ver19().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      PoolShapeInference_opset19(ctx, true);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    12,
    generated::ai_onnx::MaxPool_ver12().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      PoolShapeInference_opset19(ctx, true);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    12,
    generated::ai_onnx::Dropout_ver12().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    24,
    generated::ai_onnx::Flatten_ver24().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    23,
    generated::ai_onnx::Flatten_ver23().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    21,
    generated::ai_onnx::Flatten_ver21().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    13,
    generated::ai_onnx::Flatten_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    11,
    generated::ai_onnx::Flatten_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    1,
    generated::ai_onnx::LRN_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(MeanVarianceNormalization, 9, generated::ai_onnx::MeanVarianceNormalization_ver9());

static void convPoolShapeInference_opset1_to_11(
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
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size = checkedAdd(effective_input_size, pads[i]);
    effective_input_size = checkedAdd(effective_input_size, pads[i + kernel_shape_size]);

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions = 0;

    const int64_t stride_gap = checkedSubtract(effective_input_size, effective_kernel_shape[i]);
    if (ceil_mode == 1)
      // exact ceil division; (a + b - 1) / b would be wrong for negative a
      strided_kernel_positions =
          checkedAdd(checkedDivide(stride_gap, strides[i]), (stride_gap % strides[i] > 0 ? 1 : 0));
    else
      strided_kernel_positions = checkedDivide(stride_gap, strides[i]);

    // add in the initial position
    newdim->set_dim_value(checkedAdd(1, strided_kernel_positions));
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator_opset1_to_8() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (ctx.getNumOutputs() > 1) {
        // MaxPool with two outputs case.
        auto* output_type = ctx.getOutputType(1);
        if (output_type->value_case() == TypeProto::kTensorType ||
            output_type->value_case() == TypeProto::VALUE_NOT_SET) {
          output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
        }
      }
      convPoolShapeInference_opset1_to_11(ctx, false, true, 0, 1);
    });
  };
}

static void PoolShapeInference_opset10_to_11(InferenceContext& ctx, bool use_dilation) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* output_type = ctx.getOutputType(1);
    if (output_type->value_case() == TypeProto::kTensorType || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
      output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    }
  }
  convPoolShapeInference_opset1_to_11(ctx, use_dilation, true, 0, 1);
}

static void PoolShapeInference_opset11(InferenceContext& ctx, bool use_dilation) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* output_type = ctx.getOutputType(1);
    if (output_type->value_case() == TypeProto::kTensorType || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
      output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    }
  }
  convPoolShapeInference_opset1_to_11(ctx, use_dilation, true, 0, 1);
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    1,
    generated::ai_onnx::AveragePool_ver1().FillUsing(PoolOpSchemaGenerator_opset1_to_8()));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    7,
    generated::ai_onnx::AveragePool_ver7().FillUsing(PoolOpSchemaGenerator_opset1_to_8()));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    10,
    generated::ai_onnx::AveragePool_ver10().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      PoolShapeInference_opset10_to_11(ctx, false);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    11,
    generated::ai_onnx::AveragePool_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      PoolShapeInference_opset11(ctx, true);
    }));

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 1, generated::ai_onnx::MaxPool_ver1().FillUsing(PoolOpSchemaGenerator_opset1_to_8()));

ONNX_OPERATOR_SET_SCHEMA(MaxPool, 8, generated::ai_onnx::MaxPool_ver8().FillUsing(PoolOpSchemaGenerator_opset1_to_8()));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    10,
    generated::ai_onnx::MaxPool_ver10().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      PoolShapeInference_opset10_to_11(ctx, true);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    11,
    generated::ai_onnx::MaxPool_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      PoolShapeInference_opset10_to_11(ctx, true);
    }));

static void maxUnpoolShapeInference_opset9(InferenceContext& ctx) {
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
    9,
    generated::ai_onnx::MaxUnpool_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      maxUnpoolShapeInference_opset9(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(LpPool, 1, generated::ai_onnx::LpPool_ver1());

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator_opset2() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset1_to_11(ctx, false, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 2, generated::ai_onnx::LpPool_ver2().FillUsing(LpPoolOpSchemaGenerator_opset2()));

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator_opset11() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset1_to_11(ctx, false, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 11, generated::ai_onnx::LpPool_ver11().FillUsing(LpPoolOpSchemaGenerator_opset11()));

static std::function<void(OpSchema&)> ConvOpSchemaGenerator_opset1() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset1_to_11(ctx, true, false, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Conv, 1, generated::ai_onnx::Conv_ver1().FillUsing(ConvOpSchemaGenerator_opset1()));

static void convTransposeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);
  if (group <= 0) {
    fail_shape_inference("Attribute group must be > 0 for ConvTranspose. group=", group, ".");
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 3) {
    fail_shape_inference(
        "Input tensor must have at least 3 dimensions (N x C x D1...Dn). Got: ", input_shape.dim_size());
  }
  const auto& input_channels_dim = input_shape.dim(1);
  if (input_channels_dim.has_dim_value() && input_channels_dim.dim_value() % group != 0) {
    fail_shape_inference(
        "Input channels C must be divisible by group for ConvTranspose. C=",
        input_channels_dim.dim_value(),
        " group=",
        group,
        ".");
  }

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

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator_opset1() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { convTransposeShapeInference_opset1(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    1,
    generated::ai_onnx::ConvTranspose_ver1().FillUsing(ConvTransposeOpSchemaGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(GlobalLpPool, 1, generated::ai_onnx::GlobalLpPool_ver1());

ONNX_OPERATOR_SET_SCHEMA(BatchNormalization, 1, generated::ai_onnx::BatchNormalization_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    9,
    generated::ai_onnx::BatchNormalization_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    14,
    generated::ai_onnx::BatchNormalization_ver14().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_OPERATOR_SET_SCHEMA(InstanceNormalization, 1, generated::ai_onnx::InstanceNormalization_ver1());

ONNX_OPERATOR_SET_SCHEMA(Dropout, 1, generated::ai_onnx::Dropout_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    6,
    generated::ai_onnx::Dropout_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    7,
    generated::ai_onnx::Dropout_ver7().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    10,
    generated::ai_onnx::Dropout_ver10().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
      if (ctx.getNumOutputs() == 2) {
        updateOutputElemType(ctx, 1, TensorProto::BOOL);
        if (hasNInputShapes(ctx, 1)) {
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    6,
    generated::ai_onnx::BatchNormalization_ver6().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    1,
    generated::ai_onnx::Flatten_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    9,
    generated::ai_onnx::Flatten_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    7,
    generated::ai_onnx::BatchNormalization_ver7().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    18,
    generated::ai_onnx::GroupNormalization_ver18().SetContextDependentFunctionBodyBuilder(
        [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
          // GroupNormalization <epsilon, num_groups> (X, scale, bias) => (Y)
          auto tp = ctx.getInputType(0);
          if ((tp == nullptr) || (!tp->has_tensor_type()))
            return false;
          int64_t T = tp->tensor_type().elem_type();

          auto epsilon_attr = ctx.getAttribute("epsilon");
          float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;
          auto num_groups_attr = ctx.getAttribute("num_groups");
          if (num_groups_attr == nullptr)
            return false;
          int64_t num_groups = num_groups_attr->i();

          FunctionBuilder builder(functionProto);
          builder.Const1D("FloatEpsilon", epsilon)
              .Add("Epsilon = Cast (FloatEpsilon)", "to", T)
              .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
              .Add("C = Shape <start = 1, end = 2> (X)")
              .Const1D("NumGroups", num_groups)
              .Add("GroupSize = Div (C, NumGroups)")
              .Add("N = Shape <start = 0, end = 1> (X)") // batch size
              .Add("InstanceShape = Shape <start = 2> (X)") // data instance shape

              // NewShape = [N, num_groups, group_size, H, W, (...)]
              .Add("NewShape = Concat <axis = 0> (N, NumGroups, GroupSize, InstanceShape)")
              .Add("XReshaped = Reshape (X, NewShape)")

              // Flatten into 3D tensor: [N, num_groups, group_size x H x W (x ...)]
              .Add("Shape3D = Constant <value_ints = [0, 0, -1]> ()")
              .Add("X3D = Reshape(XReshaped, Shape3D)")

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
              .Add("Normalized = Div (Deviation, StdDev)")

              // Reshape scale and bias for broadcasting
              .Add("ScaleShape = Constant <value_ints = [1, -1, 1]> ()")
              .Add("ScaleT = Cast (scale)", "to", T)
              .Add("BiasT = Cast (bias)", "to", T)
              .Add("ScaleReshaped = Reshape (ScaleT, ScaleShape)")
              .Add("BiasReshaped = Reshape (BiasT, ScaleShape)")

              // Calculate scaled and biased output
              .Add("Scaled = Mul (ScaleReshaped, Normalized)")
              .Add("Biased = Add (Scaled, BiasReshaped)")
              .Add("Y = Reshape (Biased, XShape)");

          schema.BuildFunction(functionProto);
          return true;
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    23,
    generated::ai_onnx::Attention_ver23()
        .TypeAndShapeInferenceFunction(defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // ScaledDotProductAttention <scale, is_causal, q_num_heads, kv_numheads> (Q, K, V, attn_mask, past_key,
          // past_value) => (Y, present_key?, present_value?)
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

          if (!defs::nn::utils::AttentionAppendFunctionCausalMask(ctx, builder, false, true))
            return false;
          builder.Add("AttnBiasT = Cast (AttnBiasCausalOrNot)", "to", T1);

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
          builder
              .Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, QKHeadSize)")
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
          // entirely -inf softmaxes to NaN. Detect such rows on the additive bias
          // row-max and zero their probabilities with Where (not Mul; NaN * 0 = NaN)
          // BEFORE the P @ V contraction so 0 @ V = 0. The guard runs before the
          // mode-3 output capture so the exposed qk_matmul_output row is also zeroed,
          // consistent with Y. Mirrors the reference impl so primary == _expanded
          // bit-for-bit. In v23 the only reachable fully-masked rows come from an
          // all-False bool attn_mask (no nonpad/external cache).
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
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    24,
    generated::ai_onnx::Attention_ver24()
        .TypeAndShapeInferenceFunction(defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // ScaledDotProductAttention <scale, is_causal, q_num_heads, kv_numheads> (Q, K, V, attn_mask, past_key,
          // past_value) => (Y, present_key?, present_value?)
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
                .Add("PaddingMask4D = CastLike(PaddingMask4DFloat, AttnBiasCausalOrNot)")
                .Add("AttnBiasCausalPad = Add(AttnBiasCausalOrNot, PaddingMask4D)");
          } else {
            builder.Add("AttnBiasCausalPad = Identity(AttnBiasCausalOrNot)");
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
          builder
              .Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, QKHeadSize)")
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
        }));

} // namespace ONNX_NAMESPACE
