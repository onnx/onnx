// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/traditionalml/utils.h"

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/safe_math.h"

namespace ONNX_NAMESPACE::defs::traditionalml::utils {

void ArrayFeatureExtractorShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  if (input_ndim == 1) {
    return;
  }
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  // This operator only applies to the last dimension; thus -1
  for (int i = 0; i < input_ndim - 1; ++i) {
    *output_shape->add_dim() = input_shape.dim(i);
  }

  // value of the output's last dimension is the total amount of indices
  // set Unknown length for the last dimension if it cannot be calculated
  auto last_dim = output_shape->add_dim();
  if (hasInputShape(ctx, 1)) {
    const auto& indices_shape = getInputShape(ctx, 1);
    if (indices_shape.dim_size() > 0) {
      int64_t num_indices = 1;
      std::string single_symbolic_dim;
      for (int i = 0; i < indices_shape.dim_size(); i++) {
        if (indices_shape.dim(i).has_dim_value()) {
          if (checked_mul_overflow(num_indices, indices_shape.dim(i).dim_value(), &num_indices)) {
            fail_shape_inference("Dimension product overflow in ArrayFeatureExtractor");
          }
        } else if (indices_shape.dim(i).has_dim_param()) {
          if (single_symbolic_dim.empty()) {
            // it is possible to set symbolic dimension param if the rest dim values are all
            // value 1
            single_symbolic_dim = indices_shape.dim(i).dim_param();
          } else {
            return;
          }
        } else {
          return;
        }
      }
      if (single_symbolic_dim.empty()) {
        last_dim->set_dim_value(num_indices);
      } else if (num_indices == 1) {
        last_dim->set_dim_param(single_symbolic_dim);
      }
    }
  }
}

void CastMapShapeInference_opset1(InferenceContext& ctx) {
  auto cast_to_attr = ctx.getAttribute("cast_to");
  auto output_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (nullptr == cast_to_attr) {
    output_type->set_elem_type(TensorProto::FLOAT);
    return;
  }
  auto& cast_to = cast_to_attr->s();
  if ("TO_FLOAT" == cast_to) {
    output_type->set_elem_type(TensorProto::FLOAT);
  } else if ("TO_INT64" == cast_to) {
    output_type->set_elem_type(TensorProto::INT64);
  } else if ("TO_STRING" == cast_to) {
    output_type->set_elem_type(TensorProto::STRING);
  }
}

void CategoryMapperShapeInference_opset1(InferenceContext& ctx) {
  auto cats_int64s = ctx.getAttribute("cats_int64s");
  if (cats_int64s == nullptr) {
    fail_shape_inference("Attribute 'cats_int64s' is required.");
  }
  auto cats_strings = ctx.getAttribute("cats_strings");
  if (cats_strings == nullptr) {
    fail_shape_inference("Attribute 'cats_strings' is required.");
  }
  if (cats_int64s->ints_size() != cats_strings->strings_size()) {
    fail_shape_inference("Attributes 'cats_int64s' and 'cats_strings' are required to be the same length.");
  }
  if (nullptr == ctx.getInputType(0)) {
    return;
  }
  auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
  if (TensorProto::STRING == input_elem_type) {
    updateOutputElemType(ctx, 0, TensorProto::INT64);
  } else if (TensorProto::INT64 == input_elem_type) {
    updateOutputElemType(ctx, 0, TensorProto::STRING);
  }
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

void DictVectorizerShapeInference_opset1(InferenceContext& ctx) {
  auto input_elem_type = ctx.getInputType(0)->map_type().value_type().tensor_type().elem_type();
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_elem_type->set_elem_type(input_elem_type);
}

void LabelEncoderShapeInference_opset4(InferenceContext& ctx) {
  auto [key_type, key_length] =
      getAttributeElementTypeAndLength(ctx, {"keys_tensor", "keys_strings", "keys_int64s", "keys_floats"});
  if (key_type == TensorProto::UNDEFINED) {
    fail_shape_inference("At least one of keys_tensor, keys_strings, keys_int64s, keys_floats must be set.");
  }
  if (key_type != ctx.getInputType(0)->tensor_type().elem_type()) {
    fail_shape_inference(
        "The input type was ",
        ctx.getInputType(0)->tensor_type().elem_type(),
        " and the key type ",
        key_type,
        " are different, which is not permitted for LabelEncoders.");
  }

  auto [value_type, value_length] =
      getAttributeElementTypeAndLength(ctx, {"values_tensor", "values_strings", "values_int64s", "values_floats"});
  if (value_type == TensorProto::UNDEFINED) {
    fail_shape_inference("At least one of values_tensor, values_strings, values_int64s, values_floats must be set.");
  }
  if (value_length != key_length) {
    fail_shape_inference(
        "The number of keys ",
        key_length,
        " and the number of values ",
        value_length,
        " must be the same in the LabelEncoder.");
  }

  auto default_attr = ctx.getAttribute("default_tensor");
  if (nullptr != default_attr && default_attr->has_t() && default_attr->t().has_data_type() &&
      default_attr->t().data_type() != TensorProto_DataType_UNDEFINED) {
    auto default_tensor = default_attr->t();
    if (default_tensor.data_type() != value_type) {
      fail_shape_inference(
          "The default tensor type ",
          default_tensor.data_type(),
          " and the value type ",
          value_type,
          " must be the same in the LabelEncoder.");
    }
    if (1 != default_tensor.dims_size() || 1 != default_tensor.dims(0)) {
      fail_shape_inference("The default tensor must be a singleton 1D tensor.");
    }
  }
  // Propagate shape from input type and assign output type based on value type
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(value_type);
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

void LinearClassifierShapeInference_opset1(InferenceContext& ctx) {
  std::vector<std::string> label_strs;
  std::vector<int64_t> label_ints;

  auto labels_strings_present = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
  bool using_strings = (labels_strings_present && !label_strs.empty());

  if (!using_strings) {
    getRepeatedAttribute(ctx, "classlabels_ints", label_ints);
  }

  // Type inference
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (using_strings) {
    output_elem_type->set_elem_type(TensorProto::STRING);
  } else {
    output_elem_type->set_elem_type(TensorProto::INT64);
  }

  // second output is always of float type
  ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  // Shape/Rank inference begins

  // establish the number of classes
  std::vector<float> intercepts;
  getRepeatedAttribute(ctx, "intercepts", intercepts);
  int class_count = static_cast<int>(intercepts.size());
  if (intercepts.size() == 1 &&
      ((using_strings && label_strs.size() == 2) || (!using_strings && label_ints.size() == 2))) {
    class_count = 2;
  }

  TensorShapeProto_Dimension batch_size_dim, class_count_dim;
  class_count_dim.set_dim_value(class_count);

  if (hasNInputShapes(ctx, 1)) {
    const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    const auto input_rank = input_shape.dim_size();
    if (input_rank == 1) {
      // if input_rank is 1, batch_size is interpreted to be 1
      batch_size_dim.set_dim_value(1);
    } else if (input_rank == 2) {
      batch_size_dim = input_shape.dim(0);
    } else {
      fail_shape_inference("Input's shape should be 1D or 2D");
    }
  }

  updateOutputShape(ctx, 0, {batch_size_dim});
  updateOutputShape(ctx, 1, {batch_size_dim, class_count_dim});
}

void OneHotEncoderShapeInference_opset1(InferenceContext& ctx) {
  std::vector<int64_t> cats_int64s;
  bool has_int64s = getRepeatedAttribute(ctx, "cats_int64s", cats_int64s);
  std::vector<std::string> cats_strings;
  bool has_strings = getRepeatedAttribute(ctx, "cats_strings", cats_strings);
  if (has_int64s == has_strings) {
    fail_shape_inference("Exactly one of 'cats_*' attributes must be provided.");
  }
  // Check if input shape is available before accessing it
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < input_shape.dim_size(); i++) {
    *shape->add_dim() = input_shape.dim(i);
  }
  shape->add_dim()->set_dim_value(std::max(cats_int64s.size(), cats_strings.size()));
  updateOutputElemType(ctx, 0, TensorProto::FLOAT);
}

void SVMClassifierShapeInference_opset1(InferenceContext& ctx) {
  std::vector<std::string> label_strs;
  auto result = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
  bool using_strings = (result && !label_strs.empty());
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (using_strings) {
    output_elem_type->set_elem_type(TensorProto::STRING);
  } else {
    output_elem_type->set_elem_type(TensorProto::INT64);
  }
}

void TreeEnsembleShapeInference_opset5(InferenceContext& ctx) {
  checkInputRank(ctx, 0, 2);
  auto nodes_splits = ctx.getAttribute("nodes_splits");
  if (nullptr == nodes_splits) {
    fail_shape_inference("Attribute 'nodes_splits' is required.");
  }
  if (nodes_splits->t().dims_size() != 1) {
    fail_shape_inference("Attribute 'nodes_splits' must be 1D.");
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  // Check that input type is same as split type
  if (input_type != nodes_splits->t().data_type()) {
    fail_shape_inference(
        "Attribute 'nodes_splits' must have same type as input. Input type is ",
        input_type,
        " and attribute type is ",
        nodes_splits->t().data_type());
  }

  // Expected nodes_* length
  auto expected_length = nodes_splits->t().dims(0);
  // Validate all nodes_* attributes that are set have the same length and are 1D.
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_featureids"), expected_length, TensorProto_DataType_INT64, true);
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_hitrates"), expected_length, TensorProto_DataType_FLOAT, false);
  AssertAttributeProtoTypeAndLength(ctx.getAttribute("nodes_modes"), expected_length, TensorProto_DataType_UINT8, true);
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_truenodeids"), expected_length, TensorProto_DataType_INT64, true);
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_falsenodeids"), expected_length, TensorProto_DataType_INT64, true);
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_trueleafs"), expected_length, TensorProto_DataType_INT64, true);
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_falseleafs"), expected_length, TensorProto_DataType_INT64, true);
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_missing_value_tracks_true"), expected_length, TensorProto_DataType_INT64, false);

  // The set membership values and the splits must have the same type as the input.
  auto membership_values = ctx.getAttribute("membership_values");
  if (nullptr != membership_values && membership_values->t().data_type() != input_type) {
    fail_shape_inference(
        "Attribute 'membership_values' must have same type as input. Input type is ",
        input_type,
        " and attribute type is ",
        membership_values->t().data_type());
  }
  AssertAttributeProtoTypeAndLength(
      ctx.getAttribute("nodes_splits"), expected_length, static_cast<TensorProto_DataType>(input_type), true);

  // Validate all leaf_* attributes that are set have the same length and are 1D.
  auto leaf_targetids = ctx.getAttribute("leaf_targetids");
  auto leaf_weights = ctx.getAttribute("leaf_weights");
  if (nullptr != leaf_targetids && nullptr != leaf_weights) {
    // dims(0) is only valid for the required 1-D tensor attribute.
    if (leaf_weights->t().dims_size() != 1) {
      fail_shape_inference("Attribute 'leaf_weights' must be 1D.");
    }
    if (leaf_targetids->ints_size() != leaf_weights->t().dims(0)) {
      fail_shape_inference(
          "Attribute 'leaf_targetids' must have same length as attribute 'leaf_weights'. 'leaf_targetids' "
          "length is ",
          leaf_targetids->ints_size(),
          " and 'leaf_weights' length is ",
          leaf_weights->t().dims(0));
    }
  } else {
    fail_shape_inference("Attributes 'leaf_targetids' and 'leaf_weights' must both be set.");
  }

  // Validate weights have same type as input.
  if (leaf_weights->t().data_type() != input_type) {
    fail_shape_inference(
        "Attribute 'leaf_weights' must have same type as input. Input type is ",
        input_type,
        " and attribute type is ",
        leaf_weights->t().data_type());
  }

  checkInputRank(ctx, 0, 2);

  Dim N, E;
  unifyInputDim(ctx, 0, 0, N);
  if (nullptr != ctx.getAttribute("n_targets")) {
    unifyDim(E, ctx.getAttribute("n_targets")->i());
  }
  updateOutputElemType(ctx, 0, input_type);
  updateOutputShape(ctx, 0, {N, E});
}

void ZipMapShapeInference_opset1(InferenceContext& ctx) {
  std::vector<std::string> classlabels_strings;
  bool result = getRepeatedAttribute(ctx, "classlabels_strings", classlabels_strings);
  auto output_map_type = ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
  auto output_value_tensor_type = output_map_type->mutable_value_type()->mutable_tensor_type();
  output_value_tensor_type->set_elem_type(TensorProto::FLOAT);
  output_value_tensor_type->mutable_shape(); // Initialize to scalar
  if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() != 1 && getInputShape(ctx, 0).dim_size() != 2) {
    fail_shape_inference("ZipMap input shape should be 1D or 2D.")
  }
  if (result && !classlabels_strings.empty()) {
    output_map_type->set_key_type(TensorProto::STRING);
  }
  std::vector<int64_t> classlabels_int64s;
  result = getRepeatedAttribute(ctx, "classlabels_int64s", classlabels_int64s);
  if (result && !classlabels_int64s.empty()) {
    output_map_type->set_key_type(TensorProto::INT64);
  }
}

void LabelEncoderShapeInference_opset1(InferenceContext& ctx) {
  auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (TensorProto::STRING == input_elem_type) {
    output_elem_type->set_elem_type(TensorProto::INT64);
  } else if (TensorProto::INT64 == input_elem_type) {
    output_elem_type->set_elem_type(TensorProto::STRING);
  }
}

void TreeEnsembleClassifierShapeInference_opset1(InferenceContext& ctx) {
  std::vector<std::string> label_strs;
  auto result = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
  bool using_strings = (result && !label_strs.empty());
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (using_strings) {
    output_elem_type->set_elem_type(TensorProto::STRING);
  } else {
    output_elem_type->set_elem_type(TensorProto::INT64);
  }
}

void TreeEnsembleClassifierShapeInference_opset3(InferenceContext& ctx) {
  auto nodes_values = ctx.getAttribute("nodes_values");
  auto nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
  auto nodes_hitrates = ctx.getAttribute("nodes_hitrates");
  auto nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
  auto class_weights = ctx.getAttribute("class_weights");
  auto class_weights_as_tensor = ctx.getAttribute("class_weights_as_tensor");
  auto base_values = ctx.getAttribute("base_values");
  auto base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

  if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
    fail_shape_inference("Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
  }
  if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
    fail_shape_inference(
        "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
  }
  if (nullptr != class_weights && nullptr != class_weights_as_tensor) {
    fail_shape_inference("Only one of the attributes 'class_weights', 'class_weights_as_tensor' should be specified.");
  }
  if (nullptr != base_values && nullptr != base_values_as_tensor) {
    fail_shape_inference("Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
  }

  std::vector<std::string> classlabels_strings;
  auto result = getRepeatedAttribute(ctx, "classlabels_strings", classlabels_strings);
  bool using_strings = (result && !classlabels_strings.empty());
  if (using_strings) {
    updateOutputElemType(ctx, 0, TensorProto::STRING);
  } else {
    updateOutputElemType(ctx, 0, TensorProto::INT64);
  }
  updateOutputElemType(ctx, 1, TensorProto::FLOAT);

  checkInputRank(ctx, 0, 2);
  Dim N, E;
  unifyInputDim(ctx, 0, 0, N);

  if (using_strings) {
    unifyDim(E, classlabels_strings.size());
  } else {
    std::vector<int64_t> classlabels_int64s;
    result = getRepeatedAttribute(ctx, "classlabels_int64s", classlabels_int64s);
    if (!result || classlabels_int64s.empty()) {
      fail_shape_inference("Non of classlabels_int64s or classlabels_strings is set.");
    }
    unifyDim(E, classlabels_int64s.size());
  }
  updateOutputShape(ctx, 0, {N});
  updateOutputShape(ctx, 1, {N, E});
}

void TreeEnsembleRegressorShapeInference_opset3(InferenceContext& ctx) {
  auto nodes_values = ctx.getAttribute("nodes_values");
  auto nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
  auto nodes_hitrates = ctx.getAttribute("nodes_hitrates");
  auto nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
  auto target_weights = ctx.getAttribute("target_weights");
  auto target_weights_as_tensor = ctx.getAttribute("target_weights_as_tensor");
  auto base_values = ctx.getAttribute("base_values");
  auto base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

  if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
    fail_shape_inference("Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
  }
  if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
    fail_shape_inference(
        "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
  }
  if (nullptr != target_weights && nullptr != target_weights_as_tensor) {
    fail_shape_inference(
        "Only one of the attributes 'target_weights', 'target_weights_as_tensor' should be specified.");
  }
  if (nullptr != base_values && nullptr != base_values_as_tensor) {
    fail_shape_inference("Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
  }

  checkInputRank(ctx, 0, 2);
  Dim N, E;
  unifyInputDim(ctx, 0, 0, N);
  if (nullptr != ctx.getAttribute("n_targets")) {
    unifyDim(E, ctx.getAttribute("n_targets")->i());
  }
  updateOutputElemType(ctx, 0, TensorProto::FLOAT);
  updateOutputShape(ctx, 0, {N, E});
}

void LabelEncoderShapeInference_opset2(InferenceContext& ctx) {
  // Label encoder is one-to-one mapping.
  if (ctx.getNumInputs() != 1) {
    fail_shape_inference("Label encoder has only one input.");
  }
  if (ctx.getNumOutputs() != 1) {
    fail_shape_inference("Label encoder has only one output.");
  }

  // Load all key_* attributes.
  std::vector<std::string> keys_strings;
  bool keys_strings_result = getRepeatedAttribute(ctx, "keys_strings", keys_strings);
  std::vector<int64_t> keys_int64s;
  bool keys_int64s_result = getRepeatedAttribute(ctx, "keys_int64s", keys_int64s);
  std::vector<float> keys_floats;
  bool keys_floats_result = getRepeatedAttribute(ctx, "keys_floats", keys_floats);

  // Check if only one keys_* attribute is set.
  if (static_cast<int>(keys_strings_result) + static_cast<int>(keys_int64s_result) +
          static_cast<int>(keys_floats_result) !=
      1) {
    fail_shape_inference("Only one of keys_*'s can be set in label encoder.");
  }

  // Check if the specified keys_* matches input type.
  auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
  if (keys_strings_result && input_elem_type != TensorProto::STRING) {
    fail_shape_inference("Input type is not string tensor but key_strings is set");
  }
  if (keys_int64s_result && input_elem_type != TensorProto::INT64) {
    fail_shape_inference("Input type is not int64 tensor but keys_int64s is set");
  }
  if (keys_floats_result && input_elem_type != TensorProto::FLOAT) {
    fail_shape_inference("Input type is not float tensor but keys_floats is set");
  }

  // Load all values_* attributes.
  std::vector<std::string> values_strings;
  bool values_strings_result = getRepeatedAttribute(ctx, "values_strings", values_strings);
  std::vector<int64_t> values_int64s;
  bool values_int64s_result = getRepeatedAttribute(ctx, "values_int64s", values_int64s);
  std::vector<float> values_floats;
  bool values_floats_result = getRepeatedAttribute(ctx, "values_floats", values_floats);

  // Check if only one values_* attribute is set.
  if (static_cast<int>(values_strings_result) + static_cast<int>(values_int64s_result) +
          static_cast<int>(values_floats_result) !=
      1) {
    fail_shape_inference("Only one of values_*'s can be set in label encoder.");
  }

  // Assign output type based on the specified values_*.
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (values_strings_result)
    output_elem_type->set_elem_type(TensorProto::STRING);
  if (values_int64s_result)
    output_elem_type->set_elem_type(TensorProto::INT64);
  if (values_floats_result)
    output_elem_type->set_elem_type(TensorProto::FLOAT);

  // Input and output shapes are the same.
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

} // namespace ONNX_NAMESPACE::defs::traditionalml::utils
