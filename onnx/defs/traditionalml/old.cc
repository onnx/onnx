// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    1,
    OpSchema()
        .SetDoc(kDoc_LabelEncoder_ver1)
        .Input(0, "X", "Input data.", "T1")
        .Output(0, "Y", "Output data. If strings are input, the output values are integers, and vice versa.", "T2")
        .TypeConstraint(
            "T1",
            {types::String, types::Int64},
            "The input type must be a tensor of integers or strings, of any shape.")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64},
            "The output type will be a tensor of strings or integers, and will have the same shape as the input.")
        .Attr("classes_strings", "A list of labels.", AttributeProto::STRINGS, OPTIONAL_VALUE)
        .Attr(
            "default_int64",
            "An integer to use when an input string value is not found in the map.<br>One and only one of the "
            "'default_*' attributes must be defined.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "default_string",
            "A string to use when an input integer value is not found in the map.<br>One and only one of the "
            "'default_*' attributes must be defined.",
            AttributeProto::STRING,
            std::string("_Unused"))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (TensorProto::STRING == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::INT64);
          } else if (TensorProto::INT64 == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          }
        }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    1,
    OpSchema()
        .SetDoc(kDoc_TreeEnsembleClassifier_ver1)
        .Input(0, "X", "Input of shape [N,F]", "T1")
        .Output(0, "Y", "N, Top class for each point", "T2")
        .Output(1, "Z", "The class score for each class, for each point, a tensor of shape [N,E].", "tensor(float)")
        .TypeConstraint(
            "T1",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input type must be a tensor of a numeric type.")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* "
            "attributes is used.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Ids may restart at zero for each tree, but it not required to.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the "
            "'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and "
            "the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("class_treeids", "The id of the tree that this node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_nodeids", "node id that this weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_ids", "The index of the class list that each weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_weights", "The weight for the class in class_id.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    3,
    OpSchema()
        .SetDoc(kDoc_TreeEnsembleClassifier_ver3)
        .Input(0, "X", "Input of shape [N,F]", "T1")
        .Output(0, "Y", "N, Top class for each point", "T2")
        .Output(1, "Z", "The class score for each class, for each point, a tensor of shape [N,E].", "tensor(float)")
        .TypeConstraint(
            "T1",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input type must be a tensor of a numeric type.")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* "
            "attributes is used.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Ids may restart at zero for each tree, but it not required to.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_values_as_tensor",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates_as_tensor",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the "
            "'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and "
            "the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("class_treeids", "The id of the tree that this node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_nodeids", "node id that this weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_ids", "The index of the class list that each weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_weights", "The weight for the class in class_id.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "class_weights_as_tensor",
            "The weight for the class in class_id.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "base_values_as_tensor",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto nodes_values = ctx.getAttribute("nodes_values");
          auto nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
          auto nodes_hitrates = ctx.getAttribute("nodes_hitrates");
          auto nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
          auto class_weights = ctx.getAttribute("class_weights");
          auto class_weights_as_tensor = ctx.getAttribute("class_weights_as_tensor");
          auto base_values = ctx.getAttribute("base_values");
          auto base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

          if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
          }
          if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
          }
          if (nullptr != class_weights && nullptr != class_weights_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'class_weights', 'class_weights_as_tensor' should be specified.");
          }
          if (nullptr != base_values && nullptr != base_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
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
        }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    1,
    OpSchema()
        .SetDoc(kDoc_TreeEnsembleRegressor_ver1)
        .Input(0, "X", "Input of shape [N,F]", "T")
        .Output(0, "Y", "N classes", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Node ids must restart at zero for each tree and increase sequentially.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) "
            "or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may "
            "be left undefined and the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("target_treeids", "The id of the tree that each node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_ids", "The index of the target that each weight is for", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_weights", "The weight for each target", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'",
            AttributeProto::STRING,
            std::string("SUM"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE));

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    3,
    OpSchema()
        .SetDoc(kDoc_TreeEnsembleRegressor_ver3)
        .Input(0, "X", "Input of shape [N,F]", "T")
        .Output(0, "Y", "N classes", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Node ids must restart at zero for each tree and increase sequentially.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_values_as_tensor",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates_as_tensor",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) "
            "or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may "
            "be left undefined and the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("target_treeids", "The id of the tree that each node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_ids", "The index of the target that each weight is for", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_weights", "The weight for each target", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("target_weights_as_tensor", "The weight for each target", AttributeProto::TENSOR, OPTIONAL_VALUE)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'",
            AttributeProto::STRING,
            std::string("SUM"))
        .Attr(
            "base_values",
            "Base values for regression, added to final prediction after applying aggregate_function; the size must be "
            "the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "base_values_as_tensor",
            "Base values for regression, added to final prediction after applying aggregate_function; the size must be "
            "the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto nodes_values = ctx.getAttribute("nodes_values");
          auto nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
          auto nodes_hitrates = ctx.getAttribute("nodes_hitrates");
          auto nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
          auto target_weights = ctx.getAttribute("target_weights");
          auto target_weights_as_tensor = ctx.getAttribute("target_weights_as_tensor");
          auto base_values = ctx.getAttribute("base_values");
          auto base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

          if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
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
            fail_shape_inference(
                "Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
          }

          checkInputRank(ctx, 0, 2);
          Dim N, E;
          unifyInputDim(ctx, 0, 0, N);
          if (nullptr != ctx.getAttribute("n_targets")) {
            unifyDim(E, ctx.getAttribute("n_targets")->i());
          }
          updateOutputElemType(ctx, 0, TensorProto::FLOAT);
          updateOutputShape(ctx, 0, {N, E});
        }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    2,
    OpSchema()
        .SetDoc(kDoc_LabelEncoder_ver2)
        .Input(0, "X", "Input data. It can be either tensor or scalar.", "T1")
        .Output(0, "Y", "Output data.", "T2")
        .TypeConstraint("T1", {types::String, types::Int64, types::Float}, "The input type is a tensor of any shape.")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64, types::Float},
            "Output type is determined by the specified 'values_*' attribute.")
        .Attr(
            "keys_strings",
            "A list of strings. One and only one of 'keys_*'s should be set.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("keys_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("keys_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "values_strings",
            "A list of strings. One and only one of 'value_*'s should be set.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("values_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("values_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("default_string", "A string.", AttributeProto::STRING, std::string("_Unused"))
        .Attr("default_int64", "An integer.", AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("default_float", "A float.", AttributeProto::FLOAT, -0.f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        }));
} // namespace ONNX_NAMESPACE
#endif
