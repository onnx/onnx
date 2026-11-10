// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/tensor/utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/function.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape) {
  if (!sizes_data.empty()) {
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      auto* dim = output_shape->mutable_dim(i);
      if (sizes_data[i] > 0) {
        dim->set_dim_value(sizes_data[i]);
      }
    }
    return;
  }
}

void KeepAspectRatioHelper(
    KeepAspectRatioPolicy policy,
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& axes,
    std::vector<int64_t>& sizes_data) {
  if (policy != KeepAspectRatioPolicy::NOT_LARGER && policy != KeepAspectRatioPolicy::NOT_SMALLER) {
    return;
  }
  double scale = policy == KeepAspectRatioPolicy::NOT_LARGER ? std::numeric_limits<double>::max()
                                                             : std::numeric_limits<double>::lowest();
  std::function<double(double, double)> reduce_f;
  if (policy == KeepAspectRatioPolicy::NOT_LARGER) {
    reduce_f = [](double a, double b) { return std::min(a, b); };
  } else {
    reduce_f = [](double a, double b) { return std::max(a, b); };
  }

  bool has_unknown_dim = false;
  for (size_t i = 0; i < sizes_data.size(); i++) {
    int d = axes.empty() ? static_cast<int>(i) : static_cast<int>(axes[i]);
    if (!input_shape.dim(d).has_dim_value()) {
      has_unknown_dim = true;
      break;
    }
    double s = static_cast<double>(sizes_data[i]) / static_cast<double>(input_shape.dim(d).dim_value());
    scale = reduce_f(scale, s);
  }
  // If there's at least one unknown dim we can't infer the output shape, since it
  // will depend on the original aspect ratio of the input.
  for (size_t i = 0; i < sizes_data.size(); i++) {
    int d = axes.empty() ? static_cast<int>(i) : static_cast<int>(axes[i]);
    sizes_data[i] = has_unknown_dim
        ? -1
        : static_cast<int64_t>(std::round(scale * static_cast<double>(input_shape.dim(d).dim_value())));
  }
}

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape) {
  // Newer opsets reuse the frozen opset 7-10 helper; give this its own body when they diverge.
  resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
}

static void resizeShapeInferenceVersioned(InferenceContext& ctx, int opset_version) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = getInputShape(ctx, 0);
  auto* output_shape = getOutputShape(ctx, 0);

  bool hasScalesInput = ctx.hasInput(2);
  bool hasSizesInput = ctx.hasInput(3);

  const TensorProto* scales = 2 < ctx.getNumInputs() ? ctx.getInputData(2) : nullptr;
  std::vector<int64_t> sizes_data;
  if (3 < ctx.getNumInputs()) {
    bool found_sizes = false;
    const auto sizes_shape = getShapeInput(ctx, 3, found_sizes);
    // If sizes is an empty shape, assume it's not provided
    if (found_sizes) {
      if (sizes_shape.dim_size() == 0) {
        hasSizesInput = false;
      } else {
        for (int i = 0; i < sizes_shape.dim_size(); ++i) {
          sizes_data.push_back(sizes_shape.dim(i).dim_value());
        }
      }
    }
  }

  // If scales is an empty constant, assume it's not provided
  if (scales && ParseData<float>(scales).empty()) {
    hasScalesInput = false;
    scales = nullptr;
  }

  if ((opset_version >= 13) && (hasScalesInput + hasSizesInput != 1)) {
    fail_shape_inference("Either `sizes` or `scales` must be provided, but not both of them");
  }

  const auto* const keep_aspect_ratio_policy_attr = ctx.getAttribute("keep_aspect_ratio_policy");
  KeepAspectRatioPolicy keep_aspect_ratio_policy = KeepAspectRatioPolicy::STRETCH;
  if (keep_aspect_ratio_policy_attr && keep_aspect_ratio_policy_attr->has_s()) {
    auto str = keep_aspect_ratio_policy_attr->s();
    if (str == "stretch") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::STRETCH;
    } else if (str == "not_larger") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::NOT_LARGER;
    } else if (str == "not_smaller") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::NOT_SMALLER;
    } else {
      fail_shape_inference("Unknown value for `keep_aspect_ratio_policy`: ", str, ".");
    }
  }

  if (hasScalesInput && keep_aspect_ratio_policy != KeepAspectRatioPolicy::STRETCH) {
    fail_shape_inference(
        "Providing `scales` is incompatible with a `keep_aspect_ratio_policy` other than \"stretch\".");
  }

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

  const auto* const axes_attr = ctx.getAttribute("axes");
  size_t rank_x = input_shape.dim_size();
  std::vector<int64_t> axes;
  if (axes_attr) {
    axes = RetrieveValues<int64_t>(*axes_attr);
    checkAxesRange(axes, static_cast<int>(rank_x));
    adjustNegativeAxes(axes, static_cast<int>(rank_x));
    checkDuplicateAxes(axes, static_cast<int>(rank_x));
  }
  if (hasSizesInput) {
    if (!axes.empty()) {
      if (sizes_data.size() != axes.size()) {
        fail_shape_inference(
            "Number of elements of input 'sizes' (",
            sizes_data.size(),
            ") does not match the number of axes (",
            axes.size(),
            ").");
      }
    } else {
      // sizes_data contains scales for all axes
      if (sizes_data.size() != rank_x) {
        fail_shape_inference(
            "Number of elements of input 'sizes' (",
            sizes_data.size(),
            ") must be same as rank of input 'X' (",
            rank_x,
            ").");
      }
    }

    // Process sizes_data according to the selected policy
    KeepAspectRatioHelper(keep_aspect_ratio_policy, input_shape, axes, sizes_data);

    // If axes subset is provided, populate new sizes_data with all dims
    if (!axes.empty()) {
      std::vector<int64_t> tmp(rank_x);
      for (size_t i = 0; i < rank_x; i++) {
        tmp[i] = input_shape.dim(static_cast<int>(i)).has_dim_value() ? input_shape.dim(static_cast<int>(i)).dim_value()
                                                                      : -1;
      }
      for (size_t i = 0; i < axes.size(); i++) {
        int d = static_cast<int>(axes[i]);
        tmp[d] = sizes_data[i];
      }
      std::swap(tmp, sizes_data);
    }

    resizeShapeInferenceHelper(input_shape, sizes_data, output_shape);
  } else if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (scales->data_type() == TensorProto::FLOAT) {
      auto scales_data = ParseData<float>(scales);

      if (!axes.empty()) {
        // scales_data contains scales for a subset of axes. The rest should not be resized
        if (scales_data.size() != axes.size()) {
          fail_shape_inference(
              "Number of elements of input 'scales' (",
              scales_data.size(),
              ") does not match the number of axes (",
              axes.size(),
              ").");
        }

        std::vector<float> tmp(rank_x, 1.0f);
        for (size_t i = 0; i < axes.size(); i++) {
          int d = static_cast<int>(axes[i]);
          tmp[d] = scales_data[i];
        }
        std::swap(tmp, scales_data);
      } else {
        // scales_data contains scales for all axes
        if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
          fail_shape_inference("Number of elements of input 'scales' must be same as rank of input 'X'");
        }
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
    // If input_shape has dim_value, we calculate the scaled result
    // If input_shape doesn's have one, we leave it here
    if (input_shape.dim(i).has_dim_value()) {
      int64_t dim_value = static_cast<int64_t>(
          std::floor(static_cast<double>(input_shape.dim(i).dim_value()) * static_cast<double>(scales_data[i])));
      // If output_shape has dim_value, we validate the calculated result
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
        dim->set_dim_value(dim_value);
      } // dim->has_dim_value()
    } // input_shape.dim(i).has_dim_value()
  }
}

void oneHotShapeInference(InferenceContext& ctx, int version) {
  // Check that the node has three inputs.
  if (ctx.getNumInputs() != 3) {
    fail_type_inference("OneHot node must have three inputs.");
  }
  // Before opset 11, 'indices' were required to be non-negative; opset 11
  // relaxed this to allow negative indices that wrap around. We can only
  // enforce the older constraint when 'indices' is available as a constant.
  if (version < 11) {
    if (const TensorProto* indices_data = ctx.getInputData(0)) {
      auto is_negative = [](auto index) { return index < 0; };
      bool has_negative = false;
      if (indices_data->data_type() == TensorProto::INT64) {
        const auto indices = ParseData<int64_t>(indices_data);
        has_negative = std::any_of(indices.begin(), indices.end(), is_negative);
      } else if (indices_data->data_type() == TensorProto::INT32) {
        const auto indices = ParseData<int32_t>(indices_data);
        has_negative = std::any_of(indices.begin(), indices.end(), is_negative);
      } else if (indices_data->data_type() == TensorProto::FLOAT) {
        const auto indices = ParseData<float>(indices_data);
        has_negative = std::any_of(indices.begin(), indices.end(), is_negative);
      } else if (indices_data->data_type() == TensorProto::DOUBLE) {
        const auto indices = ParseData<double>(indices_data);
        has_negative = std::any_of(indices.begin(), indices.end(), is_negative);
      }
      if (has_negative) {
        fail_shape_inference("Input 'indices' must be non-negative for OneHot opset < 11.");
      }
    }
  }
  // Input 'depth' must be a scalar or a single-element vector.
  // TODO(ONNX): Ideally to match spec for this input only Scalar should
  // be allowed. Making this change now can affect backward
  // compatibility for this op. Since this does not seem like a good
  // justification to update version for this op, allowing both scalar
  // and 1 element vector for now. In future when version update for
  // this op is done we should only allow scalar or change the spec to
  // allow both.
  std::optional<int64_t> depth_value;
  if (hasInputShape(ctx, 1)) {
    const auto& depth_shape = getInputShape(ctx, 1);
    if (depth_shape.dim_size() != 0 && depth_shape.dim_size() != 1) {
      fail_type_inference("Input 'depth' must be a scalar or rank 1 tensor.");
    }
    // A rank-1 'depth' must hold at most one element. A zero-element tensor is
    // tolerated here (the output depth dimension is simply left unknown) rather
    // than rejected, but more than one element is an error.
    if (depth_shape.dim_size() == 1 && depth_shape.dim(0).has_dim_value() && depth_shape.dim(0).dim_value() > 1) {
      fail_type_inference("Input 'depth' must have at most one element.");
    }
    if (const TensorProto* depth_data = ctx.getInputData(1)) {
      // Read the concrete 'depth' value from the parsed data rather than
      // trusting the declared shape: the two can disagree (e.g. a scalar or
      // dynamic-rank-1 value_info backed by a zero-element initializer). Guard
      // the index so an empty parse leaves 'depth' unknown instead of reading
      // out of bounds.
      if (depth_data->data_type() == TensorProto::INT64) {
        const auto depth_data_parsed = ParseData<int64_t>(depth_data);
        if (!depth_data_parsed.empty()) {
          depth_value = depth_data_parsed[0];
        }
      } else if (depth_data->data_type() == TensorProto::INT32) {
        const auto depth_data_parsed = ParseData<int32_t>(depth_data);
        if (!depth_data_parsed.empty()) {
          depth_value = depth_data_parsed[0];
        }
      } else if (depth_data->data_type() == TensorProto::FLOAT) {
        const auto depth_data_parsed = ParseData<float>(depth_data);
        if (!depth_data_parsed.empty()) {
          depth_value = static_cast<int64_t>(depth_data_parsed[0]);
        }
      }
      // A negative 'depth' cannot describe a valid output dimension.
      if (depth_value && *depth_value < 0) {
        fail_shape_inference("Input 'depth' must be non-negative.");
      }
    }
  }
  // Input 'values' must be a two-element vector.
  if (hasInputShape(ctx, 2)) {
    const auto& values_shape = getInputShape(ctx, 2);
    if (values_shape.dim_size() != 1) {
      fail_type_inference("Input 'values' must be rank 1 tensor.");
    }
    if (values_shape.dim(0).has_dim_value() && values_shape.dim(0).dim_value() != 2) {
      fail_type_inference("Input 'values' must have exactly two elements.");
    }
  }
  // Set output type to be the same as the third input, 'values'.
  propagateElemTypeFromInputToOutput(ctx, 2, 0);
  // Set the output shape, if input 0 (indices) shape is available.
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
    int r = indices_shape.dim_size();
    if (r < 1) {
      fail_shape_inference("Indices tensor must have rank >= 1");
    }
    int out_rank = r + 1;
    int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
    if (axis < -out_rank || axis >= out_rank) {
      // 'axis' may address any slot in the output (rank out_rank = rank(indices) + 1),
      // including the newly inserted one, so the valid range is [-out_rank, out_rank-1].
      fail_shape_inference("'axis' must be in [-rank(indices)-1, rank(indices)]");
    }
    if (axis < 0) {
      axis += out_rank;
    }
    auto* output_shape = getOutputShape(ctx, 0);
    for (int i = 0; i < out_rank; ++i) {
      auto* dim = output_shape->add_dim();
      if (i < axis) {
        if (indices_shape.dim(i).has_dim_value()) {
          dim->set_dim_value(indices_shape.dim(i).dim_value());
        } else if (indices_shape.dim(i).has_dim_param()) {
          dim->set_dim_param(indices_shape.dim(i).dim_param());
        }
      } else if (i > axis) {
        if (indices_shape.dim(i - 1).has_dim_value()) {
          dim->set_dim_value(indices_shape.dim(i - 1).dim_value());
        } else if (indices_shape.dim(i - 1).has_dim_param()) {
          dim->set_dim_param(indices_shape.dim(i - 1).dim_param());
        }
      } else if (depth_value) {
        dim->set_dim_value(*depth_value);
      }
    }
  }
}
} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE::defs::tensor::utils {

void GatherNDShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto data_rank = data_shape.dim_size();
  const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  const auto indices_rank = indices_shape.dim_size();
  const int64_t batch_dims_data = getAttribute(ctx, "batch_dims", 0);
  if (data_rank < 1 || indices_rank < 1) {
    fail_shape_inference(
        "Both `data` and `indices` input tensors in GatherND op "
        "need to have rank larger than 0.");
  }
  if (batch_dims_data < 0) {
    fail_shape_inference("attribute 'batch_dims' of 'GatherND' must not be negative");
  }
  if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
    return;
  }
  const auto indices_last_dim = indices_shape.dim(indices_rank - 1).dim_value();
  if (indices_last_dim < 0) {
    fail_shape_inference("Last dimension of `indices` input tensor in GatherND op must not be negative");
  }
  const auto last_index_dimension = checkedAdd(indices_last_dim, batch_dims_data);
  if (last_index_dimension > data_rank) {
    fail_shape_inference(
        "Last dimension of `indices` input tensor in GatherND op "
        "must not be larger than the rank of `data` tensor");
  }
  auto* output_shape = getOutputShape(ctx, 0);
  for (int i = 0; i < indices_rank - 1; ++i) {
    *output_shape->add_dim() = indices_shape.dim(i);
  }
  for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
    *output_shape->add_dim() = data_shape.dim(i);
  }
}

void gridSampleShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // If there is any input shape unknown, skip the shape inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  // Grid sample input tensor indices.
  size_t const input_param = 0, grid_param = 1;

  auto const& input_shape = getInputShape(ctx, input_param);
  auto const& grid_shape = getInputShape(ctx, grid_param);

  if (input_shape.dim_size() != grid_shape.dim_size()) {
    fail_shape_inference(
        "The input tensor and grid tensor must have the same rank for GridSample. ",
        "Got input tensor rank: ",
        input_shape.dim_size(),
        ". ",
        "Got grid tensor rank: ",
        grid_shape.dim_size(),
        ". ");
  }

  int const num_dims = input_shape.dim_size();

  if (num_dims < 3) {
    fail_shape_inference(
        "The input tensor and grid tensor ranks must be >= 3. ",
        "Got input tensor and grid tensor ranks: ",
        num_dims,
        ". ");
  }
  auto const& last_dim = grid_shape.dim(num_dims - 1);
  if (last_dim.has_dim_value() && (last_dim.dim_value() != num_dims - 2)) {
    fail_shape_inference(
        "The last dimension of the grid tensor must be the rank of the grid tensor - 2. ",
        "Got grid tensor rank: ",
        num_dims,
        "Got the last dimension of the grid tensor: ",
        last_dim.dim_value(),
        ". ");
  }

  auto* output_shape = getOutputShape(ctx, 0);
  // N
  Dim& N = *output_shape->add_dim();
  // The first call sets the dimension using the dimensions from input_shape.
  unifyDim(input_shape.dim(0), N);
  // The second call checks the dimension using the dimensions from grid_shape.
  unifyDim(grid_shape.dim(0), N);
  // C
  Dim& C = *output_shape->add_dim();
  unifyDim(input_shape.dim(1), C);
  // Other Dimensions.
  for (int i = 0; i < num_dims - 2; ++i) {
    Dim& D = *output_shape->add_dim();
    unifyDim(grid_shape.dim(1 + i), D);
  }
}

void resizeShapeInference_opset18_to_19(InferenceContext& ctx) {
  resizeShapeInferenceVersioned(ctx, 19);
}

void resizeShapeInference_opset13_to_18(InferenceContext& ctx) {
  resizeShapeInferenceVersioned(ctx, 13);
}

void resizeShapeInference_opset11_to_12(InferenceContext& ctx) {
  resizeShapeInferenceVersioned(ctx, 11);
}

void resizeShapeInference_opset7_to_10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = getInputShape(ctx, 0);
  auto* output_shape = getOutputShape(ctx, 0);
  const auto* const scales = ctx.getInputData(1);

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
      const auto scales_data = ParseData<float>(scales);
      if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
        fail_shape_inference("Number of elements of input 'scales' must be same as rank of input 'X'");
      }
      resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Input 'scales' must have float element type.");
    } // nullptr != scales
  }
}

void PadShapeInference(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference needs the input data shape
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  std::vector<int64_t> axes;
  if (hasInputShape(ctx, 3)) { //'axes' input
    const auto* const axes_initializer = ctx.getInputData(3);
    if (axes_initializer == nullptr)
      return; // can't do shape inference then

    axes = ParseData<int64_t>(axes_initializer);
    checkAxesRange(axes, input_rank);
    adjustNegativeAxes(axes, input_rank);
    checkDuplicateAxes(axes, input_rank);
  } else {
    axes.resize(input_rank);
    std::iota(axes.begin(), axes.end(), 0);
  }

  auto num_axes = axes.size();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  // Populating default dims
  std::vector<TensorShapeProto_Dimension*> out_dims(input_rank);
  for (int i = 0; i < input_rank; ++i) {
    out_dims[i] = output_shape->add_dim();
  }

  // Shape Inference if
  //     1. 'pads' are available.
  // and 2. 'axes' are available, or default.
  const TensorProto* pads_initializer = ctx.getInputData(1);
  if (nullptr != pads_initializer && !axes.empty()) {
    if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
      fail_shape_inference("'pads' input must be a 1D (shape: [2 * num_axes]) tensor of type int64");
    }

    const auto pads_data = ParseData<int64_t>(pads_initializer);
    if (pads_data.size() != 2 * num_axes) {
      fail_shape_inference(
          "Pads has incorrect number of values. Expected 2 * ",
          num_axes,
          " values. Got ",
          pads_data.size(),
          " values.");
    }

    // Set default dim values
    for (int i = 0; i < input_rank; ++i) {
      const auto& input_dim = input_shape.dim(i);
      if (input_dim.has_dim_value()) {
        out_dims[i]->set_dim_value(input_dim.dim_value());
      }
    }

    for (size_t i = 0; i < num_axes; ++i) {
      auto axis = axes[i];
      const auto& input_dim = input_shape.dim(static_cast<int>(axis));
      auto& out_dim = *out_dims[axis];
      auto total_pad = checkedAdd(pads_data[i], pads_data[num_axes + i]);
      if (input_dim.has_dim_value()) {
        out_dim.set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
      } else if (total_pad == 0) {
        out_dim = input_dim;
      }
    }
  }
}

void CastDataPropagation_opset28(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset28(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CastLikeFunctionBody_opset25(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

void CastLikeShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void BitCastShapeInference_opset26(InferenceContext& ctx) {
  // Validate that input and output types have the same bit-width
  auto input_type = ctx.getInputType(0);
  if (input_type && input_type->has_tensor_type()) {
    auto input_element_type = input_type->tensor_type().elem_type();
    auto* to_attr = ctx.getAttribute("to");
    if (to_attr) {
      auto output_element_type = static_cast<int32_t>(to_attr->i());
      int64_t input_element_bit_size = ElementBitWidth(input_element_type);
      int64_t output_element_bit_size = ElementBitWidth(output_element_type);
      if (input_element_bit_size > 0 && output_element_bit_size > 0 &&
          input_element_bit_size != output_element_bit_size) {
        fail_shape_inference(
            "BitCast requires input and output types to have the same bit-width, but input type has ",
            input_element_bit_size,
            " bits and output type has ",
            output_element_bit_size,
            " bits.");
      }
    }
  }

  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);

  // Same bit-width: output shape equals input shape
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ReshapeShapeInference_opset25(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              if (checked_mul_overflow(outputProduct, input_dim_value, &outputProduct)) {
                fail_shape_inference("Dimension product overflow in Reshape");
              }
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          if (checked_mul_overflow(outputProduct, dim_value, &outputProduct)) {
            fail_shape_inference("Dimension product overflow in Reshape");
          }
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        if (checked_mul_overflow(outputProduct, dim_value, &outputProduct)) {
          fail_shape_inference("Dimension product overflow in Reshape");
        }
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          if (checked_mul_overflow(inputProduct, dataInputTensorType.shape().dim(i).dim_value(), &inputProduct)) {
            fail_shape_inference("Dimension product overflow in Reshape");
          }
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ShapeDataPropagation_opset25(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

void ShapeShapeInference_opset25(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

void SizeDataPropagation_opset25(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

void SizeShapeInference_opset25(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void ConcatDataPropagation_opset13(DataPropagationContext& ctx) {
  if (!axisIsZero(ctx)) {
    return;
  }
  TensorShapeProto tsp;
  for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
    const auto input_data = ctx.getInputData(i);
    if (input_data == nullptr) {
      return;
    }
    for (int j = 0; j < input_data->dim_size(); ++j) {
      *tsp.add_dim() = input_data->dim(j);
    }
  }
  if (tsp.dim_size() > 0) {
    ctx.addOutputData(0, std::move(tsp));
  }
}

void ConcatShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
    return;
  }

  auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axisAttr->i());
  if (axis < -rank || axis >= rank) {
    fail_shape_inference("axis must be in [-rank, rank-1].");
  }
  if (axis < 0) {
    axis += rank;
  }

  if (numInputs == 1) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
    return;
  }

  bool all_lengths_known = true;
  int64_t total_length = 0;

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int64_t i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  for (size_t i = 0; i < numInputs; i++) {
    const auto& shape = ctx.getInputType(i)->tensor_type().shape();
    if (shape.dim_size() != rank) {
      fail_shape_inference(
          "All inputs to Concat must have same rank. Input ", i, " has rank ", shape.dim_size(), " != ", rank);
    }
    for (int j = 0; j < rank; j++) {
      if (j == axis) {
        if (shape.dim(j).has_dim_value()) {
          const int64_t dim_val = shape.dim(j).dim_value();
          if (dim_val < 0) {
            fail_shape_inference("Negative dimension value on Concat axis");
          }
          if (checked_add_overflow(total_length, dim_val, &total_length)) {
            fail_shape_inference("Integer overflow computing Concat output length");
          }
        } else {
          all_lengths_known = false;
        }
      } else {
        auto& output_dim = *output_shape->mutable_dim(j);
        const auto& input_dim = shape.dim(j);
        mergeInDimensionInfo(input_dim, output_dim, j);
      }
    }
  }

  if (all_lengths_known) {
    output_shape->mutable_dim(axis)->set_dim_value(total_length);
  }
}

void SplitShapeInference_opset18(InferenceContext& ctx) {
  for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  const auto num_outputs_attr = ctx.getAttribute("num_outputs");
  if (ctx.hasInput(1) && num_outputs_attr) {
    fail_shape_inference("Both 'split' input and 'num_outputs' attribute were given");
  }
  if (ctx.hasInput(1)) { //'split' is input
    auto split_proto = ctx.getInputData(1);
    if (split_proto == nullptr) {
      // skip if split is not an initializer
      return;
    }
    split = ParseData<int64_t>(split_proto);
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else { // no value available for 'split'
    if (num_outputs_attr) {
      const auto num_outputs = num_outputs_attr->i();
      if (num_outputs < 1) {
        fail_shape_inference("Attribute `num_outputs` value cannot be lower than 1");
      }
      if (num_outputs < static_cast<int64_t>(ctx.getNumOutputs())) {
        fail_shape_inference(
            "The 'num_outputs' attribute (",
            num_outputs,
            ") must be greater or equal to the number outputs of the 'Split' node (",
            ctx.getNumOutputs(),
            ")");
      }
      if (split_dim_value % num_outputs == 0) { // tensor is evenly splittable
        int chunk_size = split_dim_value / num_outputs;
        split.resize(num_outputs, chunk_size);
      } else { // tensor needs to be split unevenly
        int chunk_size = (split_dim_value / num_outputs) + 1;
        int last_chunk_size = split_dim_value - (chunk_size * (num_outputs - 1));
        split.resize(num_outputs - 1, chunk_size);
        split.push_back(last_chunk_size);
      }
    } else {
      fail_shape_inference("Neither 'split' input nor 'num_outputs' attribute has been given");
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

static void processSliceInputs(const int64_t input_dim_size_or_value, int64_t& start, int64_t& end, int64_t step) {
  // A negative dimension is invalid: the std::clamp calls below require the
  // upper bound to be >= the lower bound, so reject the dimension with a
  // catchable error instead of invoking undefined behavior (a hardened
  // std::clamp aborts the process with SIGABRT).
  if (input_dim_size_or_value < 0) {
    fail_shape_inference("Slice input dimension must be non-negative, got ", input_dim_size_or_value);
  }
  // process step
  if (step == 0) {
    fail_shape_inference("'step' cannot be 0 for Slice");
  }
  // Empty dimension: clamp bounds are invalid when dimension size is 0,
  // so short-circuit to produce a zero-length output.
  if (input_dim_size_or_value == 0) {
    start = 0;
    end = 0;
    return;
  }
  // process start
  if (start < 0)
    start += input_dim_size_or_value;
  if (step < 0)
    start = std::clamp(start, static_cast<int64_t>(0), input_dim_size_or_value - 1);
  else
    start = std::clamp(start, static_cast<int64_t>(0), input_dim_size_or_value);
  // process end
  if (end < 0)
    end += input_dim_size_or_value;
  if (step < 0)
    end = std::clamp(end, static_cast<int64_t>(-1), input_dim_size_or_value - 1);
  else
    end = std::clamp(end, static_cast<int64_t>(0), input_dim_size_or_value);
}

void SliceDataPropagation_opset13(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  const auto starts = ctx.getInputData(1);
  const auto ends = ctx.getInputData(2);
  bool axes_specified = ctx.getNumInputs() >= 4;
  bool steps_specified = ctx.getNumInputs() >= 5;

  const TensorShapeProto* axes = nullptr;
  const TensorShapeProto* steps = nullptr;
  if (axes_specified) {
    axes = ctx.getInputData(3);
    if (axes == nullptr) {
      return;
    }
  }
  if (steps_specified) {
    steps = ctx.getInputData(4);
    if (steps == nullptr) {
      return;
    }
  }

  if (input_data == nullptr || starts == nullptr || ends == nullptr) {
    return;
  }
  if (starts->dim_size() != ends->dim_size()) {
    fail_shape_inference(
        "Input rank for starts and ends should be the same: (", starts->dim_size(), ") vs (", ends->dim_size(), ").");
  }
  // Only supports axis = 0 since the data comes from Shape
  if ((!axes_specified || (axes->dim_size() == 1 && axes->dim(0).dim_value() == 0)) && starts->dim_size() == 1 &&
      ends->dim_size() == 1) {
    auto start = starts->dim(0).dim_value();
    auto end = ends->dim(0).dim_value();
    int64_t step = 1; // Default step is 1
    if (steps_specified) {
      if (steps->dim_size() != 1) {
        return;
      }
      if (!steps->dim(0).has_dim_value()) {
        return;
      }
      step = steps->dim(0).dim_value();
    }
    processSliceInputs(input_data->dim_size(), start, end, step);

    TensorShapeProto tsp;
    if (step > 0) {
      for (int i = start; i < end; i += step) {
        *tsp.add_dim() = input_data->dim(i);
      }
    } else {
      for (int i = start; i > end; i += step) {
        *tsp.add_dim() = input_data->dim(i);
      }
    }
    if (tsp.dim_size() > 0) {
      ctx.addOutputData(0, std::move(tsp));
    }
  }
}

void SliceShapeInference_opset13(InferenceContext& ctx) {
  size_t num_inputs = ctx.getNumInputs();
  if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
    fail_type_inference("Slice op must have either three, four or five inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  // Shape Inference if
  //     1. 2nd and 3rd input data (starts, ends) are available.
  // and 2. 4th and 5th optional input (axes, steps) are either not set,
  // or set and is initializer.
  const TensorProto* startsInitializer = ctx.getInputData(1);
  const TensorProto* endsInitializer = ctx.getInputData(2);
  const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
  const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

  if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
      (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
    const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    // we can infer the output rank - it never changes
    for (int i = 0; i < input_rank; ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  // don't know data_type- can't proceed
  if (!startsInitializer->has_data_type())
    return;

  auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
    std::vector<int64_t> vec;
    if (initializer->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else if (initializer->data_type() == TensorProto::INT32) {
      const auto data = ParseData<int32_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else {
      // unaccepted data type
      fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
    }
    return vec;
  };

  std::vector<int64_t> starts = get_initializer_data(startsInitializer);
  std::vector<int64_t> ends = get_initializer_data(endsInitializer);

  if (starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing input value for starts and ends");
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();
  std::vector<int64_t> axes(starts.size());
  if (!axesInitializer) {
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = get_initializer_data(axesInitializer);
    if (axes.size() != starts.size()) {
      fail_shape_inference("Input axes has incorrect length");
    }
  }
  checkAxesRange(axes, input_rank);
  adjustNegativeAxes(axes, input_rank);
  checkDuplicateAxes(axes, input_rank);
  std::vector<int64_t> steps;
  if (!stepsInitializer) {
    steps = std::vector<int64_t>(starts.size(), 1);
  } else {
    steps = get_initializer_data(stepsInitializer);
    if (steps.size() != axes.size()) {
      fail_shape_inference("Input steps has incorrect length");
    }
  }

  for (int i = 0; i < input_rank; ++i) {
    // first update rank of output dim
    auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto& input_dim = input_shape.dim(i);
    if (input_dim.has_dim_value()) {
      output_dim->set_dim_value(input_dim.dim_value());
    } else if (input_dim.has_dim_param()) {
      output_dim->set_dim_param(input_dim.dim_param());
    }
  }

  size_t axes_size = axes.size();
  for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

    auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(static_cast<int>(axis));

    // input dim value is missing - cannot perform shape inference for
    // this axis
    if (!input_dim.has_dim_value()) {
      // Clear any previously propagated dim_param and leave this dimension "empty",
      // before moving on to the next dimension
      ctx.getOutputType(0)
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(static_cast<int>(axis))
          ->clear_dim_param();
      continue;
    }
    auto start = starts[axis_index];
    auto end = ends[axis_index];
    auto step = steps[axis_index];
    processSliceInputs(input_dim.dim_value(), start, end, step);

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
    if (temp < 0)
      temp = 0;

    // assign output value
    ctx.getOutputType(0)
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(static_cast<int>(axis))
        ->set_dim_value(temp);
  }
}

void TransposeShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

void ScatterShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ScatterElementsShapeInference_opset18(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void GatherShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  int r = data_shape.dim_size();
  if (r < 1) {
    fail_shape_inference("data tensor must have rank >= 1");
  }
  int q = indices_shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -r || axis >= r) {
    fail_shape_inference("axis must be in [-r, r-1]");
  }
  if (axis < 0) {
    axis += r;
  }
  int out_rank = q + r - 1;
  if (out_rank == 0) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  }
  for (int i = 0; i < out_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                          : // i < axis < r
        (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                    : // i - axis < q
        data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
  }
}

void GatherElementsShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // propagate indices' shape to output if it exists
  if (hasInputShape(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 1, 0);
  }
}

void SqueezeDataPropagation_opset25(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void SqueezeShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void UnsqueezeDataPropagation_opset25(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void UnsqueezeShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

bool BuildContextDependentFunctionBodySpaceToDepth(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* blocksize_attr = ctx.getAttribute("blocksize");
  if (blocksize_attr == nullptr || !blocksize_attr->has_i() || blocksize_attr->i() <= 0) {
    return false;
  }
  const auto blocksize = blocksize_attr->i();
  const auto block_area = checkedMultiply(blocksize, blocksize);
  const auto* mode_attr = ctx.getAttribute("mode");
  const std::string mode = (mode_attr != nullptr && mode_attr->has_s()) ? mode_attr->s() : "DCR";
  if (mode != "DCR" && mode != "CRD") {
    return false;
  }

  FunctionBuilder builder(functionProto);
  builder.Const1D("blocksize", blocksize)
      .Const1D("block_area", block_area)
      .Const1D("zero", int64_t{0})
      .Const1D("one", int64_t{1})
      .Const1D("two", int64_t{2})
      .Const1D("three", int64_t{3})
      .Const1D("four", int64_t{4})
      .Add("input_shape = Shape (input)")
      .Add("N = Slice (input_shape, zero, one)")
      .Add("C = Slice (input_shape, one, two)")
      .Add("H = Slice (input_shape, two, three)")
      .Add("W = Slice (input_shape, three, four)")
      .Add("H_block = Div (H, blocksize)")
      .Add("W_block = Div (W, blocksize)")
      .Add("C_block = Mul (C, block_area)")
      .Add("tmp_shape = Concat <axis = 0> (N, C, H_block, blocksize, W_block, blocksize)")
      .Add("tmp = Reshape <allowzero = 1> (input, tmp_shape)");
  if (mode == "DCR") {
    builder.Add("tmp_transposed = Transpose <perm = [0, 3, 5, 1, 2, 4]> (tmp)");
  } else {
    builder.Add("tmp_transposed = Transpose <perm = [0, 1, 3, 5, 2, 4]> (tmp)");
  }
  builder.Add("output_shape = Concat <axis = 0> (N, C_block, H_block, W_block)")
      .Add("output = Reshape <allowzero = 1> (tmp_transposed, output_shape)");
  schema.BuildFunction(functionProto);
  return true;
}

bool BuildContextDependentFunctionBodyDepthToSpace(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* blocksize_attr = ctx.getAttribute("blocksize");
  if (blocksize_attr == nullptr || !blocksize_attr->has_i() || blocksize_attr->i() <= 0) {
    return false;
  }
  const auto blocksize = blocksize_attr->i();
  const auto block_area = checkedMultiply(blocksize, blocksize);
  const auto* mode_attr = ctx.getAttribute("mode");
  const std::string mode = (mode_attr != nullptr && mode_attr->has_s()) ? mode_attr->s() : "DCR";
  if (mode != "DCR" && mode != "CRD") {
    return false;
  }

  FunctionBuilder builder(functionProto);
  builder.Const1D("blocksize", blocksize)
      .Const1D("block_area", block_area)
      .Const1D("zero", int64_t{0})
      .Const1D("one", int64_t{1})
      .Const1D("two", int64_t{2})
      .Const1D("three", int64_t{3})
      .Const1D("four", int64_t{4})
      .Add("input_shape = Shape (input)")
      .Add("N = Slice (input_shape, zero, one)")
      .Add("C = Slice (input_shape, one, two)")
      .Add("H = Slice (input_shape, two, three)")
      .Add("W = Slice (input_shape, three, four)")
      .Add("H_block = Mul (H, blocksize)")
      .Add("W_block = Mul (W, blocksize)")
      .Add("C_block = Div (C, block_area)");
  if (mode == "DCR") {
    builder.Add("tmp_shape = Concat <axis = 0> (N, blocksize, blocksize, C_block, H, W)")
        .Add("tmp = Reshape <allowzero = 1> (input, tmp_shape)")
        .Add("tmp_transposed = Transpose <perm = [0, 3, 4, 1, 5, 2]> (tmp)");
  } else {
    builder.Add("tmp_shape = Concat <axis = 0> (N, C_block, blocksize, blocksize, H, W)")
        .Add("tmp = Reshape <allowzero = 1> (input, tmp_shape)")
        .Add("tmp_transposed = Transpose <perm = [0, 1, 4, 2, 5, 3]> (tmp)");
  }
  builder.Add("output_shape = Concat <axis = 0> (N, C_block, H_block, W_block)")
      .Add("output = Reshape <allowzero = 1> (tmp_transposed, output_shape)");
  schema.BuildFunction(functionProto);
  return true;
}

void SpaceToDepthShapeInference_opset28(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      // The spatial dimensions are split into blocks, so they must divide
      // evenly. Without this check the integer divisions below silently report
      // a truncated shape for an input the runtime rejects.
      for (const int spatial_axis : {2, 3}) {
        const auto& dim = input_shape.dim(spatial_axis);
        if (dim.has_dim_value() && dim.dim_value() % blocksize != 0) {
          fail_shape_inference(
              "Input dimension ",
              spatial_axis,
              " (",
              dim.dim_value(),
              ") must be a multiple of 'blocksize' (",
              blocksize,
              ").");
        }
      }
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) * block_area,
           input_shape.dim(2) / blocksize,
           input_shape.dim(3) / blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void DepthToSpaceShapeInference_opset28(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      // The channels are redistributed into blocksize x blocksize blocks, so
      // the channel dimension must divide evenly. Without this check the
      // integer division below silently reports a truncated shape for an input
      // the runtime rejects.
      const auto& channel_dim = input_shape.dim(1);
      if (channel_dim.has_dim_value() && channel_dim.dim_value() % block_area != 0) {
        fail_shape_inference(
            "Input channel dimension (",
            channel_dim.dim_value(),
            ") must be a multiple of 'blocksize' * 'blocksize' (",
            block_area,
            ").");
      }
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void TileShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference

  // Needs at least the first input to proceed
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  const auto repeats_inputs = ctx.getInputData(1);

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (nullptr != repeats_inputs && hasNInputShapes(ctx, 2)) {
    // shape inference is possible only when 'repeats' is an initializer
    const auto& repeats_shape = ctx.getInputType(1)->tensor_type().shape();
    if (repeats_shape.dim_size() != 1 || repeats_inputs->data_type() != TensorProto::INT64) {
      fail_shape_inference("'Repeats' input must be 1D tensor of type int64");
    }

    const auto repeats_data = ParseData<int64_t>(repeats_inputs);

    if (repeats_data.size() != static_cast<size_t>(input_rank)) {
      fail_shape_inference(
          "'Repeats' input has incorrect number of values. "
          "The number of values in 'repeats' must be equal "
          "to the number of input dimensions.");
    }

    for (int i = 0; i < input_rank; ++i) {
      if (repeats_data[i] < 0) {
        fail_shape_inference("'Repeats' input must not contain negative values");
      }
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedMultiply(input_dim.dim_value(), repeats_data[i]));
      }
    }
  } else {
    // Infer output shape's rank in any case (if repeats data is not
    // available)
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

void AffineGridShapeInference_opset20(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  checkInputRank(ctx, 1, 1);

  bool found;
  TensorShapeProto size_proto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  const auto size_length = size_proto.dim_size();
  if (size_length != 4 && size_length != 5) {
    fail_shape_inference("Length of input 'size' is ", size_length, ". It must be 4 for 2D or 5 for 5D.");
  }

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& N = size_proto.dim(0);
  *output_shape->add_dim() = N;
  // const auto& C = size_proto.dim(1); // C is not used
  if (size_length == 4) {
    // 2D case: size shape (N, C, H, W), output shape (N, C, H, W, 2)
    const auto& H = size_proto.dim(2);
    const auto& W = size_proto.dim(3);
    *output_shape->add_dim() = H;
    *output_shape->add_dim() = W;
    output_shape->add_dim()->set_dim_value(2);
  } else if (size_length == 5) {
    // 3D case: size shape (N, C, D, H, W), output shape (N, C, D, H, W, 3)
    const auto& D = size_proto.dim(2);
    const auto& H = size_proto.dim(3);
    const auto& W = size_proto.dim(4);
    *output_shape->add_dim() = D;
    *output_shape->add_dim() = H;
    *output_shape->add_dim() = W;
    output_shape->add_dim()->set_dim_value(3);
  }
}

void CompressShapeInference_opset28(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto axisAttr = ctx.getAttribute("axis");
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
    int r = indices_shape.dim_size();
    if (r < 1) {
      fail_shape_inference("Indices tensor must have rank >= 1");
    }
    if (axisAttr) {
      int axis = static_cast<int>(axisAttr->i());
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
      }
      if (axis < 0) {
        axis += r;
      }
      TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      for (int i = 0; i < indices_shape.dim_size(); i++) {
        auto dim = shape->add_dim();
        if (i != axis) {
          *dim = indices_shape.dim(i);
        }
      }
    }
  }
  if (!axisAttr) {
    updateOutputShape(ctx, 0, {Dim()});
  }
}

void OneHotShapeInference_opset28(InferenceContext& ctx) {
  oneHotShapeInference(ctx, 11);
}

void IsNaNShapeInference_opset20(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void IsInfShapeInference_opset20(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void WhereShapeInference_opset16(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 3)) {
    std::vector<const TensorShapeProto*> shapes;
    shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(2)->tensor_type().shape());
    multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  }
}

void NonZeroShapeInference_opset13(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::INT64);
  TensorShapeProto output_shape;
  auto dim = output_shape.add_dim();
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& input_shape = getInputShape(ctx, 0);
    dim->set_dim_value(input_shape.dim_size());
  }
  output_shape.add_dim();
  updateOutputShape(ctx, 0, output_shape);
}

void ReverseSequenceShapeInference_opset28(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto& first_input_shape = getInputShape(ctx, 0);
  if (first_input_shape.dim_size() < 2) {
    fail_shape_inference("'input' must have rank >= 2");
  }
  auto& seq_len_input_shape = getInputShape(ctx, 1);
  if (seq_len_input_shape.dim_size() != 1) {
    fail_shape_inference("'sequence_lens' must have rank of 1");
  }

  propagateShapeFromInputToOutput(ctx, 0, 0);
}

void UniqueShapeInference_opset28(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const TypeProto* xTensorProto = ctx.getInputType(0);
  TypeProto* yTensorProto = ctx.getOutputType(0);
  TypeProto* indicesTensorProto = nullptr;
  TypeProto* inverseIndicesTensorProto = nullptr;
  TypeProto* countsTensorProto = nullptr;

  // 'indices', 'inverse_indices', and 'counts' are 1-D tensors of
  // unknown dimension.
  // Shape inference will happen even in case of empty optional outputs,
  // graph-level shape inference should not propagate the shape downstream for empty optional outputs.
  auto num_outputs = ctx.getNumOutputs();
  if (num_outputs >= 2) {
    indicesTensorProto = ctx.getOutputType(1);
    updateOutputElemType(ctx, 1, TensorProto::INT64);
    indicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  if (num_outputs >= 3) {
    inverseIndicesTensorProto = ctx.getOutputType(2);
    updateOutputElemType(ctx, 2, TensorProto::INT64);
    inverseIndicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  if (num_outputs >= 4) {
    countsTensorProto = ctx.getOutputType(3);
    updateOutputElemType(ctx, 3, TensorProto::INT64);
    countsTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    // 'axis' is not provided. Input 'X' is flattened.
    // 'Y' is a 1-D tensor of unknown dimension.
    yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  } else {
    // 'axis' is provided.
    int axis = static_cast<int>(axisAttr->i());
    if (!xTensorProto->tensor_type().has_shape()) {
      return;
    }
    const TensorShapeProto& input_shape = xTensorProto->tensor_type().shape();
    int rank = input_shape.dim_size();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank) {
      fail_shape_inference("Invalid value for attribute axis");
    }
    // 'Y' has the same shape as 'X' except in the 'axis' dimension
    // which is unknown.
    for (int i = 0; i < input_shape.dim_size(); i++) {
      auto dim = yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      if (i != axis) {
        *dim = input_shape.dim(i);
      }
    }
  }
}

void TriluShapeInference_opset14(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference needs the input data shape
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    const int rank = static_cast<int>(input_shape.dim_size());
    if (rank < 2) {
      fail_shape_inference("Input rank must be >= 2.")
    }
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CenterCropPadFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  builder.Const("k2", std::vector<int64_t>{2});

  auto axes_attr = ctx.getAttribute("axes");
  if (axes_attr) { // axes provided, need to work on a subset of dimensions
    builder.Add("axes_input = Constant <value_ints : ints = @axes>()");
    builder.Add("x_shape_alldims = Shape (input_data)").Add("x_shape = Gather (x_shape_alldims, axes_input)");
  } else { // axes not provided, assuming all dims
    builder.Add("x_shape = Shape (input_data)");
  }

  // First: Pad step
  builder.Add("padded_sh = Max(x_shape, shape)")
      .Add("pad_amount = Sub(padded_sh, x_shape)")
      .Add("pad_amount_left = Div(pad_amount, k2)")
      .Add("pad_amount_right = Sub(pad_amount, pad_amount_left)")
      .Add("pads = Concat <axis = 0> (pad_amount_left, pad_amount_right)");
  if (axes_attr)
    builder.Add("padded_input = Pad (input_data, pads, , axes_input)");
  else
    builder.Add("padded_input = Pad (input_data, pads)");

  // Second: Slice step
  if (axes_attr) {
    builder.Add("x_shape_alldims2 = Shape (padded_input)").Add("x_shape2 = Gather (x_shape_alldims2, axes_input)");
  } else {
    builder.Add("x_shape2 = Shape (padded_input)");
  }

  builder.Add("sh_diff = Sub (x_shape2, shape)")
      .Add("start_dims = Div (sh_diff, k2)")
      .Add("end_dims = Add (start_dims, shape)");
  if (axes_attr)
    builder.Add("output_data = Slice (padded_input, start_dims, end_dims, axes_input)");
  else
    builder.Add("output_data = Slice (padded_input, start_dims, end_dims)");

  schema.BuildFunction(functionProto);
  return true;
}

void CenterCropPadShapeInference_opset18(InferenceContext& ctx) {
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

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const int64_t input_rank = input_shape.dim_size();

  std::vector<int64_t> shape;
  if (cropShapeInitializer->data_type() == TensorProto::INT64) {
    const auto data = ParseData<int64_t>(cropShapeInitializer);
    shape.insert(shape.end(), data.begin(), data.end());
  } else if (cropShapeInitializer->data_type() == TensorProto::INT32) {
    const auto data = ParseData<int32_t>(cropShapeInitializer);
    shape.insert(shape.end(), data.begin(), data.end());
  } else {
    // unaccepted data type
    fail_shape_inference("`shape` only supports `int32_t` or `int64_t` inputs");
  }

  auto axes_attr = ctx.getAttribute("axes");
  std::vector<int64_t> axes;
  if (axes_attr) {
    axes = RetrieveValues<int64_t>(*axes_attr);
    checkAxesRange(axes, input_rank);
    adjustNegativeAxes(axes, input_rank);
    checkDuplicateAxes(axes, input_rank);
  } else {
    axes.resize(input_rank);
    std::iota(axes.begin(), axes.end(), 0);
  }

  if (shape.size() != axes.size()) {
    fail_shape_inference(
        "Number of elements of input 'shape' (",
        shape.size(),
        ") does not match the number of axes (",
        axes.size(),
        ").");
  }

  // Populating default dims
  std::vector<TensorShapeProto_Dimension*> out_dims(input_rank);
  auto output_shape = getOutputShape(ctx, 0);
  for (int i = 0; i < input_rank; ++i) {
    out_dims[i] = output_shape->add_dim();
    const auto& input_dim = input_shape.dim(i);
    if (input_dim.has_dim_value()) {
      out_dims[i]->set_dim_value(input_dim.dim_value());
    } else if (input_dim.has_dim_param()) {
      out_dims[i]->set_dim_param(input_dim.dim_param());
    }
  }
  int j = 0;
  for (int axis : axes) {
    out_dims[axis]->set_dim_value(shape[j++]);
  }
}

void TensorScatterShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Check that updates and past_cache have the same rank
  if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
    const auto& past_cache_shape = getInputShape(ctx, 0);
    const auto& update_shape = getInputShape(ctx, 1);
    const int past_cache_rank = static_cast<int>(past_cache_shape.dim_size());
    const int update_rank = static_cast<int>(update_shape.dim_size());

    if (past_cache_rank != update_rank) {
      fail_shape_inference(
          "past_cache and update must have the same rank. past_cache_rank=",
          past_cache_rank,
          ", update_rank=",
          update_rank);
    }

    // Check that all dimensions match except the axis dimension
    // Read axis from attribute, default is -2
    auto axis_attr = ctx.getAttribute("axis");
    int64_t axis = axis_attr ? axis_attr->i() : -2;

    // Handle negative axis
    if (axis < 0) {
      axis += past_cache_rank;
    }

    // Validate axis is within bounds
    if (axis <= 0 || axis >= past_cache_rank) {
      fail_shape_inference("axis ", axis, " is out of range for rank ", past_cache_rank);
    }
    for (int i = 0; i < past_cache_rank; ++i) {
      if (i != axis) {
        // All dimensions except axis should match
        Dim dim;
        unifyInputDim(ctx, 0, i, dim);
        unifyInputDim(ctx, 1, i, dim);
      }
    }
  }

  // Check that write_indices is of shape (batch_size,)
  if (hasInputShape(ctx, 2)) {
    checkInputRank(ctx, 2, 1);
    Dim batch_size;
    unifyInputDim(ctx, 0, 0, batch_size);
    unifyInputDim(ctx, 2, 0, batch_size);
  }

  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void CastDataPropagation_opset25(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

static void checked_mul_into(int64_t& accumulator, int64_t value) {
  if (checked_mul_overflow(accumulator, value, &accumulator)) {
    fail_shape_inference("Dimension product overflow in Reshape");
  }
}

void CastDataPropagation_opset24(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void CastDataPropagation_opset23(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void CastDataPropagation_opset21(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void CastDataPropagation_opset19(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset19(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void CastDataPropagation_opset13(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void CastShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CastLikeFunctionBody_opset24(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

void CastLikeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CastLikeFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

void CastLikeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CastLikeFunctionBody_opset21(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

void CastLikeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CastLikeFunctionBody_opset19(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

void CastLikeShapeInference_opset19(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

bool CastLikeFunctionBody_opset15(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add("output = Cast (input)", "to", static_cast<int64_t>(target_elt_type));
  schema.BuildFunction(functionProto);
  return true;
}

void CastLikeShapeInference_opset15(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void CastShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ReshapeShapeInference_opset24(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ReshapeShapeInference_opset23(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ReshapeShapeInference_opset21(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ReshapeShapeInference_opset19(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ReshapeShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape Inference if 2nd input data (the target shape) is available
  const TensorProto* targetShapeInitializer = ctx.getInputData(1);
  if (!targetShapeInitializer) {
    return;
  }
  // Make targetShape (0 -> same as originalShape, -1 -> inferred).
  // The targetShape vector represents the specified shape for output.
  std::vector<int64_t> targetShape = ParseData<int64_t>(targetShapeInitializer);
  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShape.size(), false);
  int64_t outputProduct = 1;
  for (size_t i = 0; i < targetShape.size(); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShape[i] == -1) {
      // Check if multiple -1's. If not, set negativeOneDim, marking
      // this dimension to potentially be filled in later.
      if (negativeOneDim) {
        fail_shape_inference("Target shape may not have multiple -1 dimensions");
      }
      negativeOneDim = new_dim;
    } else if (targetShape[i] == 0) {
      // Check if data input has a shape and if the index i is within
      // its bounds. If these conditions are satisfied, any dimension
      // value/param should be propagated. If dimension value cannot be
      // inferred, set the corresponding  unresolvedZeros flag to true.
      unresolvedZeros[i] = true;
      if (dataInputTensorType.has_shape()) {
        if (i >= static_cast<size_t>(dataInputTensorType.shape().dim_size())) {
          fail_shape_inference("Invalid position of 0");
        }
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          const auto& dim_value = dataInputTensorType.shape().dim(i).dim_value();
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
          unresolvedZeros[i] = false;
        } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
          const auto& dim_param = dataInputTensorType.shape().dim(i).dim_param();
          new_dim->set_dim_param(dim_param);
        }
      }
    } else if (targetShape[i] > 0) {
      // Set the dimension value to targetShape[i]
      new_dim->set_dim_value(targetShape[i]);
      checked_mul_into(outputProduct, targetShape[i]);
    } else {
      // Check if value is less than -1; fail if so
      fail_shape_inference("Invalid dimension value: ", targetShape[i]);
    }
  }

  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ReshapeShapeInference_opset5(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape Inference if 2nd input data (the target shape) is available
  const TensorProto* targetShapeInitializer = ctx.getInputData(1);
  if (!targetShapeInitializer) {
    return;
  }
  // Make targetShape (0 -> same as originalShape, -1 -> inferred).
  // The targetShape vector represents the specified shape for output.
  std::vector<int64_t> targetShape = ParseData<int64_t>(targetShapeInitializer);

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShape.size(), false);
  int64_t outputProduct = 1;
  for (int i = 0; i < static_cast<int>(targetShape.size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShape[i] == -1) {
      // Check if multiple -1's. If not, set negativeOneDim, marking
      // this dimension to potentially be filled in later.
      if (negativeOneDim) {
        fail_shape_inference("Target shape may not have multiple -1 dimensions");
      }
      negativeOneDim = new_dim;
    } else if (targetShape[i] == 0) {
      // Check if data input has a shape and if the index i is within
      // its bounds. If these conditions are satisfied, any dimension
      // value/param should be propagated. If dimension value cannot be
      // inferred, set the corresponding  unresolvedZeros flag to true.
      unresolvedZeros[i] = true;
      if (dataInputTensorType.has_shape()) {
        if (i >= dataInputTensorType.shape().dim_size()) {
          fail_shape_inference("Invalid position of 0");
        }
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          const auto& dim_value = dataInputTensorType.shape().dim(i).dim_value();
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
          unresolvedZeros[i] = false;
        } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
          const auto& dim_param = dataInputTensorType.shape().dim(i).dim_param();
          new_dim->set_dim_param(dim_param);
        }
      }
    } else if (targetShape[i] > 0) {
      // Set the dimension value to targetShape[i]
      new_dim->set_dim_value(targetShape[i]);
      checked_mul_into(outputProduct, targetShape[i]);
    } else {
      // Check if value is less than -1; fail if so
      fail_shape_inference("Invalid dimension value: ", targetShape[i]);
    }
  }

  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ShapeDataPropagation_opset24(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

void ShapeShapeInference_opset24(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

void ShapeDataPropagation_opset23(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

void ShapeShapeInference_opset23(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

// Data propagation function for Shape op
// Propagates input shape to output shape
void ShapeOp13DataPropagator(DataPropagationContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    TensorShapeProto tsp;
    tsp.CopyFrom(input_shape);
    ctx.addOutputData(0, std::move(tsp));
  }
}

void ShapeShapeInference_opset13(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    output_length->set_dim_value(ctx.getInputType(0)->tensor_type().shape().dim_size());
  }
}

void ShapeShapeInference_opset1(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    output_length->set_dim_value(ctx.getInputType(0)->tensor_type().shape().dim_size());
  }
}

void SizeDataPropagation_opset24(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

void SizeShapeInference_opset24(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void SizeDataPropagation_opset23(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

void SizeShapeInference_opset23(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void SizeShapeInference_opset1(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void ConcatShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
    return;
  }

  auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axisAttr->i());
  if (axis < -rank || axis >= rank) {
    fail_shape_inference("axis must be in [-rank, rank-1].");
  }
  if (axis < 0) {
    axis += rank;
  }

  if (numInputs == 1) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
    return;
  }

  bool all_lengths_known = true;
  int64_t total_length = 0;

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int64_t i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  for (size_t i = 0; i < numInputs; i++) {
    const auto& shape = ctx.getInputType(i)->tensor_type().shape();
    if (shape.dim_size() != rank) {
      fail_shape_inference(
          "All inputs to Concat must have same rank. Input ", i, " has rank ", shape.dim_size(), " != ", rank);
    }
    for (int j = 0; j < rank; j++) {
      if (j == axis) {
        if (shape.dim(j).has_dim_value()) {
          const int64_t dim_val = shape.dim(j).dim_value();
          if (dim_val < 0) {
            fail_shape_inference("Negative dimension value on Concat axis");
          }
          if (checked_add_overflow(total_length, dim_val, &total_length)) {
            fail_shape_inference("Integer overflow computing Concat output length");
          }
        } else {
          all_lengths_known = false;
        }
      } else {
        auto& output_dim = *output_shape->mutable_dim(j);
        const auto& input_dim = shape.dim(j);
        mergeInDimensionInfo(input_dim, output_dim, j);
      }
    }
  }

  if (all_lengths_known) {
    output_shape->mutable_dim(axis)->set_dim_value(total_length);
  }
}

void SplitShapeInference_opset11(InferenceContext& ctx) {
  if (ctx.getNumOutputs() == 0) {
    fail_shape_inference("Split must have at least one output");
  }
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  if (getRepeatedAttribute(ctx, "split", split)) {
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else {
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim_value % num_outputs != 0) {
      fail_shape_inference("The input is not evenly splittable");
    }
    int chunk_size = split_dim_value / num_outputs;
    for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

void SplitShapeInference_opset13(InferenceContext& ctx) {
  if (ctx.getNumOutputs() == 0) {
    fail_shape_inference("Split must have at least one output");
  }
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  size_t num_inputs = ctx.getNumInputs();
  if ((num_inputs == 2) && ctx.getInputType(1)) { //'split' is input
    auto split_proto = ctx.getInputData(1);
    if (split_proto == nullptr) {
      // skip if split is not an initializer
      return;
    }
    split = ParseData<int64_t>(split_proto);
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else { // no value available for 'split'
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim_value % num_outputs != 0) {
      fail_shape_inference("The input is not evenly splittable");
    }
    int chunk_size = split_dim_value / num_outputs;
    split.reserve(ctx.getNumOutputs());
    for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

void SliceShapeInference_opset11(InferenceContext& ctx) {
  size_t num_inputs = ctx.getNumInputs();
  if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
    fail_type_inference("Slice op must have either three, four or five inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  // Shape Inference if
  //     1. 2nd and 3rd input data (starts, ends) are available.
  // and 2. 4th and 5th optional input (axes, steps) are either not set,
  // or set and is initializer.
  const TensorProto* startsInitializer = ctx.getInputData(1);
  const TensorProto* endsInitializer = ctx.getInputData(2);
  const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
  const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

  if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
      (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
    const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    // we can infer the output rank - it never changes
    for (int i = 0; i < input_rank; ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  // don't know data_type- can't proceed
  if (!startsInitializer->has_data_type())
    return;

  auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
    std::vector<int64_t> vec;
    if (initializer->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else if (initializer->data_type() == TensorProto::INT32) {
      const auto data = ParseData<int32_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else {
      // unaccepted data type
      fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
    }
    return vec;
  };

  std::vector<int64_t> starts = get_initializer_data(startsInitializer);
  std::vector<int64_t> ends = get_initializer_data(endsInitializer);

  if (starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing input value for starts and ends");
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();
  std::vector<int64_t> axes(starts.size());
  if (!axesInitializer) {
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = get_initializer_data(axesInitializer);
    if (axes.size() != starts.size()) {
      fail_shape_inference("Input axes has incorrect length");
    }
  }

  std::vector<int64_t> steps;
  if (!stepsInitializer) {
    steps = std::vector<int64_t>(starts.size(), 1);
  } else {
    steps = get_initializer_data(stepsInitializer);
    if (steps.size() != axes.size()) {
      fail_shape_inference("Input steps has incorrect length");
    }
  }

  for (int i = 0; i < input_rank; ++i) {
    // first update rank of output dim
    auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto& input_dim = input_shape.dim(i);
    if (input_dim.has_dim_value()) {
      output_dim->set_dim_value(input_dim.dim_value());
    } else if (input_dim.has_dim_param()) {
      output_dim->set_dim_param(input_dim.dim_param());
    }
  }

  std::unordered_set<int64_t> unique_axes;
  size_t axes_size = axes.size();
  for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

    if (axis >= static_cast<int64_t>(input_rank) || axis < 0) {
      fail_shape_inference("Input axes has invalid data");
    }

    if (unique_axes.find(axis) != unique_axes.end()) {
      fail_shape_inference("'axes' has duplicates");
    }

    unique_axes.insert(axis);

    auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(static_cast<int>(axis));

    // input dim value is missing - cannot perform shape inference for
    // this axis
    if (!input_dim.has_dim_value()) {
      // Clear any previously propagated dim_param and leave this
      // dimension "empty", before moving on to the next dimension
      ctx.getOutputType(0)
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(static_cast<int>(axis))
          ->clear_dim_param();
      continue;
    }

    const auto input_dim_value = input_dim.dim_value();

    // A negative dimension is invalid: the std::clamp calls below require the
    // upper bound to be >= the lower bound, so reject the dimension with a
    // catchable error instead of invoking undefined behavior (a hardened
    // std::clamp aborts the process with SIGABRT).
    if (input_dim_value < 0) {
      fail_shape_inference("Slice input dimension must be non-negative, got ", input_dim_value);
    }

    // Empty dimension: clamp bounds are invalid when dimension size is 0,
    // so short-circuit to produce a zero-length output.
    if (input_dim_value == 0) {
      ctx.getOutputType(0)
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(static_cast<int>(axis))
          ->set_dim_value(0);
      continue;
    }

    // process step
    auto step = steps[axis_index];
    if (step == 0) {
      fail_shape_inference("'step' cannot be 0");
    }

    // process start
    auto start = starts[axis_index];
    if (start < 0)
      start += input_dim_value;
    if (step < 0)
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value - 1);
    else
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value);

    // process end
    auto end = ends[axis_index];
    if (end < static_cast<int64_t>(0))
      end += input_dim_value;
    if (step < static_cast<int64_t>(0))
      end = std::clamp(end, static_cast<int64_t>(-1), input_dim_value);
    else
      end = std::clamp(end, static_cast<int64_t>(0), input_dim_value);

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
    if (temp < 0)
      temp = 0;

    // assign output value
    ctx.getOutputType(0)
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(static_cast<int>(axis))
        ->set_dim_value(temp);
  }
}

void TransposeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

void TransposeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

void TransposeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

void TransposeShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

void TransposeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

void ScatterNDShapeInference_opset16(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ScatterNDShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ScatterNDShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ScatterElementsShapeInference_opset16(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ScatterElementsShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ScatterElementsShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void GatherShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  int r = data_shape.dim_size();
  if (r < 1) {
    fail_shape_inference("data tensor must have rank >= 1");
  }
  int q = indices_shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -r || axis >= r) {
    fail_shape_inference("axis must be in [-r, r-1]");
  }
  if (axis < 0) {
    axis += r;
  }
  int out_rank = q + r - 1;
  if (out_rank == 0) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  }
  for (int i = 0; i < out_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                          : // i < axis < r
        (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                    : // i - axis < q
        data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
  }
}

void GatherElementsShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // propagate indices' shape to output if it exists
  if (hasInputShape(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 1, 0);
  }
}

void SqueezeDataPropagation_opset24(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void SqueezeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void SqueezeDataPropagation_opset23(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void SqueezeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void SqueezeDataPropagation_opset21(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void SqueezeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void SqueezeDataPropagation_opset13(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void SqueezeShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void SqueezeShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    for (int i = 0; i < input_ndim; ++i) {
      if (!input_shape.dim(i).has_dim_value()) {
        return;
      }
      if (input_shape.dim(i).dim_value() == 1) {
        axes.push_back(i);
      }
    }
  }

  std::transform(axes.begin(), axes.end(), axes.begin(), [&](int64_t axis) -> int64_t {
    return axis < 0 ? axis + input_ndim : axis;
  });

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < input_ndim; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void UnsqueezeDataPropagation_opset24(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void UnsqueezeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

void UnsqueezeDataPropagation_opset23(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void UnsqueezeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

void UnsqueezeDataPropagation_opset21(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void UnsqueezeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

void UnsqueezeDataPropagation_opset13(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

void UnsqueezeShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

void UnsqueezeShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    return;
  }

  // validate 'axes' for duplicate entries
  std::unordered_set<int64_t> unique_values;
  for (const auto val : axes) {
    if (unique_values.find(val) != unique_values.end()) {
      fail_shape_inference("'axes' attribute must not contain any duplicates");
    }
    unique_values.insert(val);
  }

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  for (auto& axe : axes) {
    if (axe < -output_ndim || axe >= output_ndim) {
      fail_shape_inference("values in 'axes' are beyond the bounds of the computed output shape");
    }
    if (axe < 0) {
      axe += output_ndim;
    }
  }

  // sort after correcting negative axes values (if any) in the previous
  // step
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

void SpaceToDepthShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) * block_area,
           input_shape.dim(2) / blocksize,
           input_shape.dim(3) / blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void SpaceToDepthShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) * block_area,
           input_shape.dim(2) / blocksize,
           input_shape.dim(3) / blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void DepthToSpaceShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void DepthToSpaceShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void TileShapeInference_opset6(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference

  // Needs at least the first input to proceed
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  const auto repeats_inputs = ctx.getInputData(1);

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (nullptr != repeats_inputs && hasNInputShapes(ctx, 2)) {
    // shape inference is possible only when 'repeats' is an initializer
    const auto& repeats_shape = ctx.getInputType(1)->tensor_type().shape();
    if (repeats_shape.dim_size() != 1 || repeats_inputs->data_type() != TensorProto::INT64) {
      fail_shape_inference("'Repeats' input must be 1D tensor of type int64");
    }

    const auto repeats_data = ParseData<int64_t>(repeats_inputs);

    if (repeats_data.size() != static_cast<size_t>(input_rank)) {
      fail_shape_inference(
          "'Repeats' input has incorrect number of values. "
          "The number of values in 'repeats' must be equal "
          "to the number of input dimensions.");
    }

    for (int i = 0; i < input_rank; ++i) {
      if (repeats_data[i] < 0) {
        fail_shape_inference("'Repeats' input must not contain negative values");
      }
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedMultiply(input_dim.dim_value(), repeats_data[i]));
      }
    }
  } else {
    // Infer output shape's rank in any case (if repeats data is not
    // available)
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

void IsNaNShapeInference_opset9(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void IsNaNShapeInference_opset13(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void IsInfShapeInference_opset10(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void NonZeroShapeInference_opset9(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::INT64);
  TensorShapeProto output_shape;
  auto dim = output_shape.add_dim();
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& input_shape = getInputShape(ctx, 0);
    dim->set_dim_value(input_shape.dim_size());
  }
  output_shape.add_dim();
  updateOutputShape(ctx, 0, output_shape);
}

void GatherNDShapeInference_opset12(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (!hasNInputShapes(ctx, 2)) {
    // cannot proceed with shape or rank inference
    return;
  }

  const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto data_rank = data_shape.dim_size();

  const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  const auto indices_rank = indices_shape.dim_size();

  int64_t batch_dims_data = getAttribute(ctx, "batch_dims", 0);
  if (data_rank < 1 || indices_rank < 1) {
    fail_shape_inference(
        "Both `data` and `indices` input tensors in GatherND op "
        "need to have rank larger than 0.");
  }
  if (batch_dims_data < 0) {
    fail_shape_inference("attribute 'batch_dims' of 'GatherND' must not be negative");
  }

  // cannot ascertain if the input shapes are valid if shape of
  // `indices` is missing last dimension value so return at this point
  if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
    return;
  }

  const auto indices_last_dim = indices_shape.dim(indices_rank - 1).dim_value();
  if (indices_last_dim < 0) {
    fail_shape_inference("Last dimension of `indices` input tensor in GatherND op must not be negative");
  }
  const auto last_index_dimension = checkedAdd(indices_last_dim, batch_dims_data);

  if (last_index_dimension > data_rank) {
    fail_shape_inference(
        "Last dimension of `indices` input tensor in GatherND op "
        "must not be larger than the rank of `data` tensor");
  }

  for (int i = 0; i < indices_rank - 1; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
  }

  for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
  }
}

void PadShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference needs the input data shape
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  // Infer output shape if 'pads' tensor is available
  const auto pads_initializer = ctx.getInputData(1);
  if (nullptr != pads_initializer) {
    if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
      fail_shape_inference("'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");
    }

    const auto pads_data = ParseData<int64_t>(pads_initializer);
    if (pads_data.size() != static_cast<size_t>(2 * input_rank)) {
      fail_shape_inference("Pads has incorrect number of values");
    }

    auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < input_rank; ++i) {
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      const auto total_pad = checkedAdd(pads_data[i], pads_data[i + input_rank]);
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
      } else if (total_pad == 0) {
        *output_dim = input_dim;
      }
    }
  } else {
    // Infer output shapes' rank in any case
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

void CastShapeInference_opset6(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void ConcatShapeInference_opset4(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
    return;
  }

  auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axisAttr->i());
  if (rank <= axis) {
    fail_shape_inference("rank must be greater than axis");
  }
  if (axis < 0) {
    return;
  }

  bool all_lengths_known = true;
  int64_t total_length = 0;

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int64_t i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  for (size_t i = 0; i < numInputs; i++) {
    const auto& shape = ctx.getInputType(i)->tensor_type().shape();
    if (shape.dim_size() != rank) {
      fail_shape_inference("All inputs to Concat must have same rank");
    }
    for (int j = 0; j < rank; j++) {
      if (j == axis) {
        if (shape.dim(j).has_dim_value()) {
          const int64_t dim_val = shape.dim(j).dim_value();
          if (dim_val < 0) {
            fail_shape_inference("Negative dimension value on Concat axis");
          }
          if (checked_add_overflow(total_length, dim_val, &total_length)) {
            fail_shape_inference("Integer overflow computing Concat output length");
          }
        } else {
          all_lengths_known = false;
        }
      } else {
        auto& output_dim = *output_shape->mutable_dim(j);
        const auto& input_dim = shape.dim(j);
        mergeInDimensionInfo(input_dim, output_dim, j);
      }
    }
  }

  if (all_lengths_known) {
    output_shape->mutable_dim(axis)->set_dim_value(total_length);
  }
}

void TileShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Only rank of output can be inferred. We can do better if second
  // input is a constant, but this requires extending InferenceContext
  // interface to get values of constant inputs.
}

void UpsampleShapeInference_opset7(InferenceContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const auto& input_shape = getInputShape(ctx, 0);
  auto output_shape = getOutputShape(ctx, 0);
  const auto scales = ctx.getAttribute("scales");

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
    if (scales->type() == AttributeProto_AttributeType_FLOATS) {
      const std::vector<float> scales_data(scales->floats().begin(), scales->floats().end());
      if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
        fail_shape_inference("Number of elements of attribute 'scales' must be same as rank of input 'X'");
      }
      resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Attribute 'scales' must have floats type.");
    } // scales->type() == float
  } else {
    fail_shape_inference("Attribute 'scales' is required.");
  } // nullptr != scales
}

void SliceShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  if (!getRepeatedAttribute(ctx, "starts", starts) || !getRepeatedAttribute(ctx, "ends", ends) ||
      starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing attribute value for starts and ends");
  }

  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(i);
    }
  } else if (axes.size() != starts.size()) {
    fail_shape_inference("Attribute axes has incorrect length");
  } else if (!std::is_sorted(axes.begin(), axes.end())) {
    // TODO(ONNX) support shape inference for unsorted axes
    return;
  }

  auto is_negative = [](int64_t index) { return index < 0; };
  if (std::any_of(axes.begin(), axes.end(), is_negative)) {
    // Negative axes were not explicitly discussed in the spec before opset-10.
    // Hence, they are officially not part of the spec, but some models/runtimes may use them.
    // So we perform simple rank inference in this case.
    for (size_t i = 0; static_cast<int64_t>(i) < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (size_t i = 0, j = 0; static_cast<int64_t>(i) < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
    auto newdim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    if (j < axes.size() && static_cast<size_t>(axes[j]) == i) {
      // There's a lot of potential behaviors. For now just
      // handle some simple cases.
      const auto& dim = ctx.getInputType(0)->tensor_type().shape().dim(i);
      if (dim.has_dim_value()) {
        auto dim_value = dim.dim_value();
        if (starts[j] < 0) {
          starts[j] += dim_value;
        }
        if (ends[j] < 0) {
          ends[j] += dim_value;
        }
        if (starts[j] >= 0 && ends[j] >= 0) {
          auto newval = std::min(dim_value, ends[j]) - starts[j];
          if (newval >= 0) {
            newdim->set_dim_value(newval);
          }
        }
      }
      ++j;
    } else {
      *newdim = ctx.getInputType(0)->tensor_type().shape().dim(i);
    }
  }
}

void SliceShapeInference_opset10(InferenceContext& ctx) {
  size_t num_inputs = ctx.getNumInputs();
  if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
    fail_type_inference("Slice op must have either three, four or five inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  // Shape Inference if
  //     1. 2nd and 3rd input data (starts, ends) are available.
  // and 2. 4th and 5th optional input (axes, steps) are either not set,
  // or set and is initializer.
  const TensorProto* startsInitializer = ctx.getInputData(1);
  const TensorProto* endsInitializer = ctx.getInputData(2);
  const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
  const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

  if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
      (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
    const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    // we can infer the output rank - it never changes
    for (int i = 0; i < input_rank; ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  // don't know data_type- can't proceed
  if (!startsInitializer->has_data_type())
    return;

  auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
    std::vector<int64_t> vec;
    if (initializer->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else if (initializer->data_type() == TensorProto::INT32) {
      const auto data = ParseData<int32_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else {
      // unaccepted data type
      fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
    }
    return vec;
  };

  std::vector<int64_t> starts = get_initializer_data(startsInitializer);
  std::vector<int64_t> ends = get_initializer_data(endsInitializer);

  if (starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing input value for starts and ends");
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();
  std::vector<int64_t> axes(starts.size());
  if (!axesInitializer) {
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = get_initializer_data(axesInitializer);
    if (axes.size() != starts.size()) {
      fail_shape_inference("Input axes has incorrect length");
    }
  }

  std::vector<int64_t> steps;
  if (!stepsInitializer) {
    steps = std::vector<int64_t>(starts.size(), 1);
  } else {
    steps = get_initializer_data(stepsInitializer);
    if (steps.size() != axes.size()) {
      fail_shape_inference("Input steps has incorrect length");
    }
  }

  for (int i = 0; i < input_rank; ++i) {
    // first update rank of output dim
    auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto& input_dim = input_shape.dim(i);
    if (input_dim.has_dim_value()) {
      output_dim->set_dim_value(input_dim.dim_value());
    } else if (input_dim.has_dim_param()) {
      output_dim->set_dim_param(input_dim.dim_param());
    }
  }

  std::unordered_set<int64_t> unique_axes;
  size_t axes_size = axes.size();
  for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

    if (axis >= static_cast<int64_t>(input_rank) || axis < 0) {
      fail_shape_inference("Input axes has invalid data");
    }

    if (unique_axes.find(axis) != unique_axes.end()) {
      fail_shape_inference("'axes' has duplicates");
    }

    unique_axes.insert(axis);
    auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(axis);

    // input dim value is missing - cannot perform shape inference for
    // this axis
    if (!input_dim.has_dim_value())
      continue;

    const auto input_dim_value = input_dim.dim_value();

    // A negative dimension is invalid: the std::clamp calls below require the
    // upper bound to be >= the lower bound, so reject the dimension with a
    // catchable error instead of invoking undefined behavior (a hardened
    // std::clamp aborts the process with SIGABRT).
    if (input_dim_value < 0) {
      fail_shape_inference("Slice input dimension must be non-negative, got ", input_dim_value);
    }

    // Empty dimension: clamp bounds are invalid when dimension size is 0,
    // so short-circuit to produce a zero-length output.
    if (input_dim_value == 0) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(0);
      continue;
    }

    // process step
    auto step = steps[axis_index];
    if (step == 0) {
      fail_shape_inference("'step' cannot be 0");
    }

    // process start
    auto start = starts[axis_index];
    if (start < 0)
      start += input_dim_value;
    if (step < 0)
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value - 1);
    else
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value);

    // process end
    auto end = ends[axis_index];
    if (end < static_cast<int64_t>(0))
      end += input_dim_value;
    if (step < static_cast<int64_t>(0))
      end = std::clamp(end, static_cast<int64_t>(-1), input_dim_value);
    else
      end = std::clamp(end, static_cast<int64_t>(0), input_dim_value);

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
    if (temp < 0)
      temp = 0;

    // assign output value
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(temp);
  }
}

void ScatterShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void DepthToSpaceShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void GatherShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  int r = data_shape.dim_size();
  if (r < 1) {
    fail_shape_inference("data tensor must have rank >= 1");
  }
  int q = indices_shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -r || axis >= r) {
    fail_shape_inference("axis must be in [-r, r-1]");
  }
  if (axis < 0) {
    axis += r;
  }
  int out_rank = q + r - 1;

  if (out_rank == 0) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  }
  for (int i = 0; i < out_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                          : // i < axis < r
        (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                    : // i - axis < q
        data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
  }
}

void SqueezeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    for (int i = 0; i < input_ndim; ++i) {
      if (!input_shape.dim(i).has_dim_value()) {
        return;
      }
      if (input_shape.dim(i).dim_value() == 1) {
        axes.push_back(i);
      }
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0, j = 0; i < input_shape.dim_size(); ++i) {
    if (static_cast<size_t>(j) < axes.size() && axes[j] == i) {
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
      ++j;
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

void UnsqueezeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    return;
  }
  std::sort(axes.begin(), axes.end());

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  int j = 0;
  for (int i = 0; i < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

void OneHotShapeInference_opset9(InferenceContext& ctx) {
  oneHotShapeInference(ctx, 9);
}

void SplitShapeInference_opset2(InferenceContext& ctx) {
  if (ctx.getNumOutputs() == 0) {
    fail_shape_inference("Split must have at least one output");
  }
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  // Previously Split-2 does not mention how to deal with negative axis
  // However, there is an existing test onnx/backend/test/data/pytorch-converted/test_GLU
  // using Split-2 with negative axis and it is hard to be regenerated.
  // To compromise, handle negative axis for Split-2 here.
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  if (getRepeatedAttribute(ctx, "split", split)) {
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else {
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim_value % num_outputs != 0) {
      fail_shape_inference("The input is not evenly splittable");
    }
    int chunk_size = split_dim_value / num_outputs;
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

void PadShapeInference_opset2(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto& input_shape = ctx.getInputType(0)->tensor_type().shape();

  std::vector<int64_t> pads;
  if (!getRepeatedAttribute(ctx, "pads", pads)) {
    fail_shape_inference("Attribute value for pads is required");
  }
  if (pads.size() != static_cast<size_t>(input_shape.dim_size() * 2)) {
    fail_shape_inference("Attribute pads has incorrect length");
    ;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    auto newdim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto total_pad = checkedAdd(pads[i], pads[input_shape.dim_size() + i]);
    if (ctx.getInputType(0)->tensor_type().shape().dim(i).has_dim_value()) {
      newdim->set_dim_value(checkedAdd(ctx.getInputType(0)->tensor_type().shape().dim(i).dim_value(), total_pad));
    } else if (total_pad == 0) {
      *newdim = input_shape.dim(i);
    }
  }
}

void GatherNDShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (!hasNInputShapes(ctx, 2)) {
    // cannot proceed with shape or rank inference
    return;
  }

  const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto data_rank = data_shape.dim_size();

  const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  const auto indices_rank = indices_shape.dim_size();

  if (data_rank < 1 || indices_rank < 1) {
    fail_shape_inference(
        "Both `data` and `indices` input tensors in GatherND op "
        "need to have rank larger than 0.");
  }

  // cannot ascertain if the input shapes are valid if shape of
  // `indices` is missing last dimension value so return at this point
  if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
    return;
  }

  const auto last_index_dimension = indices_shape.dim(indices_rank - 1).dim_value();

  if (last_index_dimension < 0) {
    fail_shape_inference("Last dimension of `indices` input tensor in GatherND op must not be negative");
  }
  if (last_index_dimension > data_rank) {
    fail_shape_inference(
        "Last dimension of `indices` input tensor in GatherND op "
        "must not be larger than the rank of `data` tensor");
  }

  for (int i = 0; i < indices_rank - 1; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
  }

  for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
  }
}

void WhereShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 3)) {
    std::vector<const TensorShapeProto*> shapes;
    shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(2)->tensor_type().shape());
    multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  }
}

void PadShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference needs the input data shape
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  // Infer output shape if 'pads' tensor is available
  const auto pads_initializer = ctx.getInputData(1);
  if (nullptr != pads_initializer) {
    if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
      fail_shape_inference("'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");
    }

    const auto pads_data = ParseData<int64_t>(pads_initializer);
    if (pads_data.size() != static_cast<size_t>(2 * input_rank)) {
      fail_shape_inference("Pads has incorrect number of values");
    }

    auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < input_rank; ++i) {
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      const auto total_pad = checkedAdd(pads_data[i], pads_data[i + input_rank]);
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
      } else if (total_pad == 0) {
        *output_dim = input_dim;
      }
    }
  } else {
    // Infer output shapes' rank in any case
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

void ReshapeShapeInference_opset14(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

void ShapeDataPropagation_opset21(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

void ShapeShapeInference_opset21(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

void ShapeDataPropagation_opset19(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

void ShapeShapeInference_opset19(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

void ShapeDataPropagation_opset15(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

void ShapeShapeInference_opset15(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

void SizeDataPropagation_opset21(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

void SizeShapeInference_opset21(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void SizeDataPropagation_opset19(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

void SizeShapeInference_opset19(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void SizeDataPropagation_opset13(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

void SizeShapeInference_opset13(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

void CompressShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto axisAttr = ctx.getAttribute("axis");
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
    int r = indices_shape.dim_size();
    if (r < 1) {
      fail_shape_inference("Indices tensor must have rank >= 1");
    }
    if (axisAttr) {
      int axis = static_cast<int>(axisAttr->i());
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
      }
      if (axis < 0) {
        axis += r;
      }
      TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      for (int i = 0; i < indices_shape.dim_size(); i++) {
        auto dim = shape->add_dim();
        if (i != axis) {
          *dim = indices_shape.dim(i);
        }
      }
    }
  }
  if (!axisAttr) {
    updateOutputShape(ctx, 0, {Dim()});
  }
}

void OneHotShapeInference_opset11(InferenceContext& ctx) {
  oneHotShapeInference(ctx, 11);
}

void ReverseSequenceShapeInference_opset10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto& first_input_shape = getInputShape(ctx, 0);
  if (first_input_shape.dim_size() < 2) {
    fail_shape_inference("'input' must have rank >= 2");
  }
  auto& seq_len_input_shape = getInputShape(ctx, 1);
  if (seq_len_input_shape.dim_size() != 1) {
    fail_shape_inference("'sequence_lens' must have rank of 1");
  }

  propagateShapeFromInputToOutput(ctx, 0, 0);
}

void UniqueShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const TypeProto* xTensorProto = ctx.getInputType(0);
  TypeProto* yTensorProto = ctx.getOutputType(0);
  TypeProto* indicesTensorProto = nullptr;
  TypeProto* inverseIndicesTensorProto = nullptr;
  TypeProto* countsTensorProto = nullptr;

  auto num_outputs = ctx.getNumOutputs();
  if (num_outputs >= 2) {
    indicesTensorProto = ctx.getOutputType(1);
    updateOutputElemType(ctx, 1, TensorProto::INT64);
    indicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  if (num_outputs >= 3) {
    inverseIndicesTensorProto = ctx.getOutputType(2);
    updateOutputElemType(ctx, 2, TensorProto::INT64);
    inverseIndicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  if (num_outputs >= 4) {
    countsTensorProto = ctx.getOutputType(3);
    updateOutputElemType(ctx, 3, TensorProto::INT64);
    countsTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  } else {
    int axis = static_cast<int>(axisAttr->i());
    if (!xTensorProto->tensor_type().has_shape()) {
      return;
    }
    const TensorShapeProto& input_shape = xTensorProto->tensor_type().shape();
    int rank = input_shape.dim_size();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank) {
      fail_shape_inference("Invalid value for attribute axis");
    }
    for (int i = 0; i < input_shape.dim_size(); i++) {
      auto dim = yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      if (i != axis) {
        *dim = input_shape.dim(i);
      }
    }
  }
}

} // namespace ONNX_NAMESPACE::defs::tensor::utils
