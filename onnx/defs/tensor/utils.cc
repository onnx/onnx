/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/tensor/utils.h"

namespace ONNX_NAMESPACE {
void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape) {
  if (!sizes_data.empty()) {
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      auto* dim = output_shape->mutable_dim(i);
      dim->set_dim_value(sizes_data[i]);
    }
    return;
  }
}

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape) {
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    auto* dim = output_shape->mutable_dim(i);
    // If input_shape has dim_value, we calculate the scaled result
    // If input_shape doesn's have one, we leave it here
    if (input_shape.dim(i).has_dim_value()) {
      int64_t dim_value =
          static_cast<int64_t>(std::floor(static_cast<float>(input_shape.dim(i).dim_value()) * scales_data[i]));
      // If output_shape has dim_value, we validate the caculated result
      // If output_shape doesn's have one, we set it to the scaled result
      if (dim->has_dim_value()) {
        if (static_cast<int64_t>(dim->dim_value()) != dim_value) {
          fail_shape_inference(
              "Dimension value inferred (",
              dim_value,
              ") is not equal to the existing dim value (",
              dim->dim_value(),
              ").");
        }
      } else {
        dim->set_dim_value(static_cast<int64_t>(dim_value));
      } // dim->has_dim_value()
    } // input_shape.dim(i).has_dim_value()
  }
}

void resizeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = getInputShape(ctx, 0);
  auto* output_shape = getOutputShape(ctx, 0);
  const TensorProto* scales = 2 < ctx.getNumInputs() ? ctx.getInputData(2) : nullptr;

  if (output_shape->dim_size() > 0) {
    if (output_shape->dim_size() != input_shape.dim_size()) {
      fail_shape_inference(
          "Ranks inferred (",
          input_shape.dim_size(),
          ") is not equal to the existing rank value (",
          output_shape->dim_size(),
          ").");
    }
  } else { // Infer the rank of output anyway
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      output_shape->add_dim();
    }
  }

  if (ctx.getNumInputs() == 4) {
    const auto* sizes = ctx.getInputData(3);
    if (nullptr != sizes) {
      if (sizes->data_type() == TensorProto::INT64) {
        const auto sizes_data = ParseData<int64_t>(sizes);
        if (sizes_data.size() != static_cast<size_t>(input_shape.dim_size())) {
          fail_shape_inference("Number of elements of input 'sizes' must be same as rank of input 'X'");
        }
        resizeShapeInferenceHelper(input_shape, sizes_data, output_shape);
      } else {
        fail_shape_inference("Input 'sizes' must have int64 element type.");
      }
    }
  } else if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (scales->data_type() == TensorProto::FLOAT) {
      const auto& scales_data = ParseData<float>(scales);
      if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
        fail_shape_inference("Number of elements of input 'scales' must be same as rank of input 'X'");
      }
      resizeShapeInferenceHelper(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Input 'scales' must have float element type.");
    }
  } // nullptr != scales
}

void resizeShapeInferenceHelper_opset7_to_10(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape) {
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    auto* dim = output_shape->mutable_dim(i);
    // If input_shape has dim_value, we caculate the scaled result
    // If input_shape doesn's have one, we leave it here
    if (input_shape.dim(i).has_dim_value()) {
      int64_t dim_value =
          static_cast<int64_t>(std::floor(static_cast<float>(input_shape.dim(i).dim_value()) * scales_data[i]));
      // If output_shape has dim_value, we validate the caculated result
      // If output_shape doesn's have one, we set it to the scaled result
      if (dim->has_dim_value()) {
        if (static_cast<int64_t>(dim->dim_value()) != dim_value) {
          fail_shape_inference(
              "Dimension value inferred (",
              dim_value,
              ") is not equal to the existing dim value (",
              dim->dim_value(),
              ").");
        }
      } else {
        dim->set_dim_value(static_cast<int64_t>(dim_value));
      } // dim->has_dim_value()
    } // input_shape.dim(i).has_dim_value()
  }
}

void resizeShapeInference_opset7_to_10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = getInputShape(ctx, 0);
  auto* output_shape = getOutputShape(ctx, 0);
  const auto scales = ctx.getInputData(1);

  if (output_shape->dim_size() > 0) {
    if (output_shape->dim_size() != input_shape.dim_size()) {
      fail_shape_inference(
          "Ranks inferred (",
          input_shape.dim_size(),
          ") is not equal to the existing rank value (",
          output_shape->dim_size(),
          ").");
    }
  } else { // Infer the rank of output anyway
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      output_shape->add_dim();
    }
  }

  if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (scales->data_type() == TensorProto::FLOAT) {
      const auto& scales_data = ParseData<float>(scales);
      if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
        fail_shape_inference("Number of elements of input 'scales' must be same as rank of input 'X'");
      }
      resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Input 'scales' must have float element type.");
    } // nullptr != scales
  }
}

} // namespace ONNX_NAMESPACE
