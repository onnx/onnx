// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

static void LabelEncoderShapeInference_opset1(InferenceContext& ctx) {
  auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
  auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
  if (TensorProto::STRING == input_elem_type) {
    output_elem_type->set_elem_type(TensorProto::INT64);
  } else if (TensorProto::INT64 == input_elem_type) {
    output_elem_type->set_elem_type(TensorProto::STRING);
  }
}

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    1,
    generated::ai_onnx_ml::LabelEncoder_ver1().TypeAndShapeInferenceFunction(LabelEncoderShapeInference_opset1));

static void TreeEnsembleClassifierShapeInference_opset1(InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    1,
    generated::ai_onnx_ml::TreeEnsembleClassifier_ver1().TypeAndShapeInferenceFunction(
        TreeEnsembleClassifierShapeInference_opset1));

static void TreeEnsembleClassifierShapeInference_opset3(InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    3,
    generated::ai_onnx_ml::TreeEnsembleClassifier_ver3().TypeAndShapeInferenceFunction(
        TreeEnsembleClassifierShapeInference_opset3));

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleRegressor, 1, generated::ai_onnx_ml::TreeEnsembleRegressor_ver1());

static void TreeEnsembleRegressorShapeInference_opset3(InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    3,
    generated::ai_onnx_ml::TreeEnsembleRegressor_ver3().TypeAndShapeInferenceFunction(
        TreeEnsembleRegressorShapeInference_opset3));

static void LabelEncoderShapeInference_opset2(InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    2,
    generated::ai_onnx_ml::LabelEncoder_ver2().TypeAndShapeInferenceFunction(LabelEncoderShapeInference_opset2));
} // namespace ONNX_NAMESPACE
#endif
