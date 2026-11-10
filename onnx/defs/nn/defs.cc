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

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    22,
    generated::ai_onnx::AveragePool_ver22().TypeAndShapeInferenceFunction(
        defs::nn::utils::AveragePoolShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    22,
    generated::ai_onnx::MaxPool_ver22().TypeAndShapeInferenceFunction(defs::nn::utils::MaxPoolShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    22,
    generated::ai_onnx::MaxUnpool_ver22().TypeAndShapeInferenceFunction(defs::nn::utils::maxUnpoolShapeInference));

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::nn::utils::LpPoolShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 22, generated::ai_onnx::LpPool_ver22().FillUsing(LpPoolOpSchemaGenerator()));

static std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::nn::utils::roiPoolTypeShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(MaxRoiPool, 22, generated::ai_onnx::MaxRoiPool_ver22().FillUsing(RoiPoolOpSchemaGenerator()));

static std::function<void(OpSchema&)> ConvOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::nn::utils::ConvShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(Conv, 22, generated::ai_onnx::Conv_ver22().FillUsing(ConvOpSchemaGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    QLinearConv,
    10,
    generated::ai_onnx::QLinearConv_ver10().TypeAndShapeInferenceFunction(
        defs::nn::utils::QLinearConvShapeInference_opset10));

ONNX_OPERATOR_SET_SCHEMA(
    ConvInteger,
    10,
    generated::ai_onnx::ConvInteger_ver10().TypeAndShapeInferenceFunction(
        defs::nn::utils::ConvIntegerShapeInference_opset10));

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::nn::utils::convTransposeShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    22,
    generated::ai_onnx::ConvTranspose_ver22().FillUsing(ConvTransposeOpSchemaGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    22,
    generated::ai_onnx::DeformConv_ver22().TypeAndShapeInferenceFunction(
        defs::nn::utils::DeformConvShapeInference_opset22));

static std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::nn::utils::globalPoolTypeShapeInference); };
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
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(defs::nn::utils::globalPoolTypeShapeInference); };
}

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    22,
    generated::ai_onnx::GlobalLpPool_ver22().FillUsing(GlobalLpPoolingOpSchemaGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    15,
    generated::ai_onnx::BatchNormalization_ver15().TypeAndShapeInferenceFunction(
        defs::nn::utils::BatchNormalizationShapeInference_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    22,
    generated::ai_onnx::InstanceNormalization_ver22().TypeAndShapeInferenceFunction(
        propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    22,
    generated::ai_onnx::LpNormalization_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    22,
    generated::ai_onnx::Dropout_ver22().TypeAndShapeInferenceFunction(defs::nn::utils::DropoutShapeInference_opset22));

ONNX_OPERATOR_SET_SCHEMA(
    Shrink,
    9,
    generated::ai_onnx::Shrink_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    25,
    generated::ai_onnx::Flatten_ver25().TypeAndShapeInferenceFunction(defs::nn::utils::FlattenShapeInference_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    13,
    generated::ai_onnx::LRN_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    TfIdfVectorizer,
    9,
    generated::ai_onnx::TfIdfVectorizer_ver9().TypeAndShapeInferenceFunction(
        defs::nn::utils::TfIdfVectorizerShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(MeanVarianceNormalization, 13, generated::ai_onnx::MeanVarianceNormalization_ver13());

ONNX_OPERATOR_SET_SCHEMA(
    Col2Im,
    18,
    generated::ai_onnx::Col2Im_ver18().TypeAndShapeInferenceFunction(defs::nn::utils::col2imShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    LayerNormalization,
    17,
    generated::ai_onnx::LayerNormalization_ver17()
        .SetContextDependentFunctionBodyBuilder(
            defs::nn::utils::BuildContextDependentFunctionBodyLayerNormalizationVer17,
            17)
        .SetContextDependentFunctionBodyBuilder(
            defs::nn::utils::BuildContextDependentFunctionBodyLayerNormalizationVer18,
            18)
        .TypeAndShapeInferenceFunction(defs::nn::utils::LayerNormalizationShapeInference_opset17));

ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    21,
    generated::ai_onnx::GroupNormalization_ver21()
        .TypeAndShapeInferenceFunction(defs::nn::utils::GroupNormalizationShapeInference)
        .SetContextDependentFunctionBodyBuilder(defs::nn::utils::GroupNormalizationFunctionBody_opset21));

ONNX_OPERATOR_SET_SCHEMA(
    RMSNormalization,
    23,
    generated::ai_onnx::RMSNormalization_ver23()
        .TypeAndShapeInferenceFunction(defs::nn::utils::RMSNormalizationShapeInference_opset23)
        .SetContextDependentFunctionBodyBuilder(defs::nn::utils::RMSNormalizationFunctionBody_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    RotaryEmbedding,
    23,
    generated::ai_onnx::RotaryEmbedding_ver23()
        .TypeAndShapeInferenceFunction(defs::nn::utils::RotaryEmbeddingShapeInference_opset23)
        .SetContextDependentFunctionBodyBuilder(defs::nn::utils::RotaryEmbeddingFunctionBody_opset23));

ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    25,
    generated::ai_onnx::Attention_ver25()
        .TypeAndShapeInferenceFunction(defs::nn::utils::Attention25Inference)
        .SetContextDependentFunctionBodyBuilder(defs::nn::utils::AttentionFunctionBody_opset25));

ONNX_OPERATOR_SET_SCHEMA(
    CausalConvWithState,
    27,
    generated::ai_onnx::CausalConvWithState_ver27()
        .TypeAndShapeInferenceFunction(defs::nn::utils::CausalConvWithStateShapeInference_opset27)
        .SetContextDependentFunctionBodyBuilder(defs::nn::utils::CausalConvWithStateFunctionBody_opset27));

ONNX_OPERATOR_SET_SCHEMA(
    LinearAttention,
    27,
    generated::ai_onnx::LinearAttention_ver27()
        .SetContextDependentFunctionBodyBuilder(defs::nn::utils::LinearAttentionFunctionBody_opset27)
        .TypeAndShapeInferenceFunction(defs::nn::utils::LinearAttentionShapeInference_opset27));
} // namespace ONNX_NAMESPACE
