// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
// Kept for backward compatibility of the public API.
ONNX_API void convTransposeShapeInference(InferenceContext& ctx) {
  defs::nn::utils::convTransposeShapeInference(ctx);
}

// Kept for backward compatibility of the public API.
ONNX_API void globalPoolTypeShapeInference(InferenceContext& ctx) {
  defs::nn::utils::globalPoolTypeShapeInference(ctx);
}

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

} // namespace ONNX_NAMESPACE
