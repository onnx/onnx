// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/traditionalml/utils.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

ONNX_ML_OPERATOR_SET_SCHEMA(
    ArrayFeatureExtractor,
    1,
    generated::ai_onnx_ml::ArrayFeatureExtractor_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    Binarizer,
    1,
    generated::ai_onnx_ml::Binarizer_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    CastMap,
    1,
    generated::ai_onnx_ml::CastMap_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    CategoryMapper,
    1,
    generated::ai_onnx_ml::CategoryMapper_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    DictVectorizer,
    1,
    generated::ai_onnx_ml::DictVectorizer_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto input_elem_type = ctx.getInputType(0)->map_type().value_type().tensor_type().elem_type();
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      output_elem_type->set_elem_type(input_elem_type);
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(FeatureVectorizer, 1, generated::ai_onnx_ml::FeatureVectorizer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(Imputer, 1, generated::ai_onnx_ml::Imputer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    4,
    generated::ai_onnx_ml::LabelEncoder_ver4().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        fail_shape_inference(
            "At least one of values_tensor, values_strings, values_int64s, values_floats must be set.");
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearClassifier,
    1,
    generated::ai_onnx_ml::LinearClassifier_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(LinearRegressor, 1, generated::ai_onnx_ml::LinearRegressor_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(Normalizer, 1, generated::ai_onnx_ml::Normalizer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(
    OneHotEncoder,
    1,
    generated::ai_onnx_ml::OneHotEncoder_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(Scaler, 1, generated::ai_onnx_ml::Scaler_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMClassifier,
    1,
    generated::ai_onnx_ml::SVMClassifier_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      std::vector<std::string> label_strs;
      auto result = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
      bool using_strings = (result && !label_strs.empty());
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      if (using_strings) {
        output_elem_type->set_elem_type(TensorProto::STRING);
      } else {
        output_elem_type->set_elem_type(TensorProto::INT64);
      }
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(SVMRegressor, 1, generated::ai_onnx_ml::SVMRegressor_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleClassifier, 5, generated::ai_onnx_ml::TreeEnsembleClassifier_ver5());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleRegressor, 5, generated::ai_onnx_ml::TreeEnsembleRegressor_ver5());

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsemble,
    5,
    generated::ai_onnx_ml::TreeEnsemble_ver5().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_modes"), expected_length, TensorProto_DataType_UINT8, true);
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
    }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    ZipMap,
    1,
    generated::ai_onnx_ml::ZipMap_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    }));

} // namespace ONNX_NAMESPACE
#endif
