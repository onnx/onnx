// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/traditionalml/utils.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

ONNX_ML_OPERATOR_SET_SCHEMA(
    ArrayFeatureExtractor,
    1,
    OpSchema()
        .SetDoc(kDoc_ArrayFeatureExtractor_ver1)
        .Input(0, "X", "Data to be selected", "T")
        .Input(1, "Y", "The indices, based on 0 as the first index of any dimension.", "tensor(int64)")
        .Output(0, "Z", "Selected output data as an array", "T")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        })
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32, types::String},
            "The input must be a tensor of a numeric type or string. The output will be of the same tensor type."));

ONNX_ML_OPERATOR_SET_SCHEMA(
    Binarizer,
    1,
    OpSchema()
        .SetDoc(kDoc_Binarizer_ver1)
        .Input(0, "X", "Data to be binarized", "T")
        .Output(0, "Y", "Binarized output data", "T")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input must be a tensor of a numeric type. The output will be of the same tensor type.")
        .Attr("threshold", "Values greater than this are mapped to 1, others to 0.", AttributeProto::FLOAT, 0.f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    CastMap,
    1,
    OpSchema()
        .SetDoc(kDoc_CastMap_ver1)
        .Input(0, "X", "The input map that is to be cast to a tensor", "T1")
        .Output(0, "Y", "A tensor representing the same data as the input map, ordered by their keys", "T2")
        .TypeConstraint(
            "T1",
            {types::Map<TensorProto::INT64>("string"), types::Map<TensorProto::INT64>("float")},
            "The input must be an integer map to either string or float.")
        .TypeConstraint(
            "T2",
            {types::String, types::Float, types::Int64},
            "The output is a 1-D tensor of string, float, or integer.")
        .Attr(
            "cast_to",
            "A string indicating the desired element type of the output tensor, one of 'TO_FLOAT', 'TO_STRING', "
            "'TO_INT64'.",
            AttributeProto::STRING,
            std::string("TO_FLOAT"))
        .Attr(
            "map_form",
            "Indicates whether to only output as many values as are in the input (dense), or position the input based "
            "on using the key of the map as the index of the output (sparse).<br>One of 'DENSE', 'SPARSE'.",
            AttributeProto::STRING,
            std::string("DENSE"))
        .Attr(
            "max_map",
            "If the value of map_form is 'SPARSE,' this attribute indicates the total length of the output tensor.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    OpSchema()
        .SetDoc(kDoc_CategoryMapper_ver1)
        .Input(0, "X", "Input data", "T1")
        .Output(0, "Y", "Output data. If strings are input, the output values are integers, and vice versa.", "T2")
        .TypeConstraint(
            "T1",
            {types::String, types::Int64},
            "The input must be a tensor of strings or integers, either [N,C] or [C].")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64},
            "The output is a tensor of strings or integers. Its shape will be the same as the input shape.")
        .Attr(
            "cats_strings",
            "The strings of the map. This sequence must be the same length as the 'cats_int64s' sequence",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "cats_int64s",
            "The integers of the map. This sequence must be the same length as the 'cats_strings' sequence.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "default_string",
            "A string to use when an input integer value is not found in the map.<br>One and only one of the "
            "'default_*' attributes must be defined.",
            AttributeProto::STRING,
            std::string("_Unused"))
        .Attr(
            "default_int64",
            "An integer to use when an input string value is not found in the map.<br>One and only one of the "
            "'default_*' attributes must be defined.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    OpSchema()
        .SetDoc(kDoc_DictVectorizer_ver1)
        .Input(0, "X", "A dictionary.", "T1")
        .Output(0, "Y", "A 1-D tensor holding values from the input dictionary.", "T2")
        .TypeConstraint(
            "T1",
            {types::Map<TensorProto::STRING>("int64"),
             types::Map<TensorProto::INT64>("string"),
             types::Map<TensorProto::INT64>("float"),
             types::Map<TensorProto::INT64>("double"),
             types::Map<TensorProto::STRING>("float"),
             types::Map<TensorProto::STRING>("double")},
            "The input must be a map from strings or integers to either strings or a numeric type. The key and value "
            "types cannot be the same.")
        .TypeConstraint(
            "T2",
            {types::Int64, types::Float, types::Double, types::String},
            "The output will be a tensor of the value type of the input map. It's shape will be [1,C], where C is the "
            "length of the input dictionary.")
        .Attr(
            "string_vocabulary",
            "A string vocabulary array.<br>One and only one of the vocabularies must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "int64_vocabulary",
            "An integer vocabulary array.<br>One and only one of the vocabularies must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)->map_type().value_type().tensor_type().elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(input_elem_type);
        }));

ONNX_ML_OPERATOR_SET_SCHEMA(
    FeatureVectorizer,
    1,
    OpSchema()
        .SetDoc(kDoc_FeatureVectorizer_ver1)
        .Input(0, "X", "An ordered collection of tensors, all with the same element type.", "T1", OpSchema::Variadic)
        .Output(0, "Y", "The output array, elements ordered as the inputs.", "tensor(float)")
        .TypeConstraint(
            "T1",
            {types::Int32, types::Int64, types::Float, types::Double},
            "The input type must be a tensor of a numeric type.")
        .Attr("inputdimensions", "The size of each input in the input list", AttributeProto::INTS, OPTIONAL_VALUE));

ONNX_ML_OPERATOR_SET_SCHEMA(
    Imputer,
    1,
    OpSchema()
        .SetDoc(kDoc_Imputer_ver1)
        .Input(0, "X", "Data to be processed.", "T")
        .Output(0, "Y", "Imputed output data", "T")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input type must be a tensor of a numeric type, either [N,C] or [C]. The output type will be of the "
            "same tensor type and shape.")
        .Attr("imputed_value_floats", "Value(s) to change to", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("replaced_value_float", "A value that needs replacing.", AttributeProto::FLOAT, 0.f)
        .Attr("imputed_value_int64s", "Value(s) to change to.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("replaced_value_int64", "A value that needs replacing.", AttributeProto::INT, static_cast<int64_t>(0)));

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    4,
    OpSchema()
        .SetDoc(kDoc_LabelEncoder_ver4)
        .Input(0, "X", "Input data. It must have the same element type as the keys_* attribute set.", "T1")
        .Output(0, "Y", "Output data. This tensor's element type is based on the values_* attribute set.", "T2")
        .TypeConstraint(
            "T1",
            {types::String, types::Int64, types::Float, types::Int32, types::Int16, types::Double},
            "The input type is a tensor of any shape.")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64, types::Float, types::Int32, types::Int16, types::Double},
            "Output type is determined by the specified 'values_*' attribute.")
        .Attr(
            "keys_tensor",
            "Keys encoded as a 1D tensor. One and only one of 'keys_*'s should be set.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr("keys_strings", "A list of strings.", AttributeProto::STRINGS, OPTIONAL_VALUE)
        .Attr("keys_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("keys_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "values_tensor",
            "Values encoded as a 1D tensor. One and only one of 'values_*'s should be set.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr("values_strings", "A list of strings.", AttributeProto::STRINGS, OPTIONAL_VALUE)
        .Attr("values_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("values_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("default_string", "A string.", AttributeProto::STRING, std::string("_Unused"))
        .Attr("default_int64", "An integer.", AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("default_float", "A float.", AttributeProto::FLOAT, -0.f)
        .Attr(
            "default_tensor",
            "A default tensor. {\"_Unused\"} if values_* has string type, {-1} if values_* has integral type, and "
            "{-0.f} if values_* has float type.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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

          auto [value_type, value_length] = getAttributeElementTypeAndLength(
              ctx, {"values_tensor", "values_strings", "values_int64s", "values_floats"});
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
    OpSchema()
        .SetDoc(kDoc_LinearClassifier_ver1)
        .Input(0, "X", "Data to be classified.", "T1")
        .Output(0, "Y", "Classification outputs (one class per example).", "T2")
        .Output(1, "Z", "Classification scores ([N,E] - one score for each class and example", "tensor(float)")
        .TypeConstraint(
            "T1",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input must be a tensor of a numeric type, and of shape [N,C] or [C]. In the latter case, it will be "
            "treated as [1,C]")
        .TypeConstraint("T2", {types::String, types::Int64}, "The output will be a tensor of strings or integers.")
        .Attr("coefficients", "A collection of weights of the model(s).", AttributeProto::FLOATS)
        .Attr("intercepts", "A collection of intercepts.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "multi_class",
            "Indicates whether to do OvR or multinomial (0=OvR is the default).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "classlabels_strings",
            "Class labels when using string labels. One and only one 'classlabels' attribute must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_ints",
            "Class labels when using integer labels. One and only one 'classlabels' attribute must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the scores vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' "
            "'SOFTMAX_ZERO,' or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearRegressor,
    1,
    OpSchema()
        .SetDoc(kDoc_LinearRegressor_ver1)
        .Input(0, "X", "Data to be regressed.", "T")
        .Output(0, "Y", "Regression outputs (one per target, per example).", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input must be a tensor of a numeric type.")
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the regression output vector.<br>One of 'NONE,' 'SOFTMAX,' "
            "'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr("coefficients", "Weights of the model(s).", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("intercepts", "Weights of the intercepts, if used.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "targets",
            "The total number of regression targets, 1 if not defined.",
            AttributeProto::INT,
            static_cast<int64_t>(1)));

ONNX_ML_OPERATOR_SET_SCHEMA(
    Normalizer,
    1,
    OpSchema()
        .SetDoc(kDoc_Normalizer_ver1)
        .Input(0, "X", "Data to be encoded, a tensor of shape [N,C] or [C]", "T")
        .Output(0, "Y", "Encoded output data", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input must be a tensor of a numeric type.")
        .Attr("norm", "One of 'MAX,' 'L1,' 'L2'", AttributeProto::STRING, std::string("MAX")));

ONNX_ML_OPERATOR_SET_SCHEMA(
    OneHotEncoder,
    1,
    OpSchema()
        .SetDoc(kDoc_OneHotEncoder_ver1)
        .Input(0, "X", "Data to be encoded.", "T")
        .Output(0, "Y", "Encoded output data, having one more dimension than X.", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::String, types::Int64, types::Int32, types::Float, types::Double},
            "The input must be a tensor of a numeric type.")
        .Attr(
            "cats_int64s",
            "List of categories, ints.<br>One and only one of the 'cats_*' attributes must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "cats_strings",
            "List of categories, strings.<br>One and only one of the 'cats_*' attributes must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "zeros",
            "If true and category is not present, will return all zeros; if false and a category if not found, the "
            "operator will fail.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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

ONNX_ML_OPERATOR_SET_SCHEMA(
    Scaler,
    1,
    OpSchema()
        .SetDoc(kDoc_Scaler_ver1)
        .Input(0, "X", "Data to be scaled.", "T")
        .Output(0, "Y", "Scaled output data.", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input must be a tensor of a numeric type.")
        .Attr(
            "offset",
            "First, offset by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it "
            "applies to all features, regardless of dimension count.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "scale",
            "Second, multiply by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it "
            "applies to all features, regardless of dimension count.<br>Must be same length as 'offset'",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE));

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMClassifier,
    1,
    OpSchema()
        .SetDoc(kDoc_SVMClassifier_ver1)
        .Input(0, "X", "Data to be classified.", "T1")
        .Output(0, "Y", "Classification outputs (one class per example).", "T2")
        .Output(
            1,
            "Z",
            "Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for "
            "each class, otherwise they are raw scores.",
            "tensor(float)")
        .TypeConstraint(
            "T1",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input must be a tensor of a numeric type, either [C] or [N,C].")
        .TypeConstraint(
            "T2",
            {types::String, types::Int64},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* "
            "attributes is used. Its size will match the batch size of the input.")
        .Attr(
            "kernel_type",
            "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.",
            AttributeProto::STRING,
            std::string("LINEAR"))
        .Attr(
            "kernel_params",
            "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr("vectors_per_class", "", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("support_vectors", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("coefficients", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("prob_a", "First set of probability coefficients.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "prob_b",
            "Second set of probability coefficients. This array must be same size as prob_a.<br>If these are provided "
            "then output Z are probability estimates, otherwise they are raw scores.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr("rho", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_ints",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::INTS,
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
    SVMRegressor,
    1,
    OpSchema()
        .SetDoc(kDoc_SVMRegressor_ver1)
        .Input(0, "X", "Data to be regressed.", "T")
        .Output(0, "Y", "Regression outputs (one score per target per example).", "tensor(float)")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Int64, types::Int32},
            "The input type must be a tensor of a numeric type, either [C] or [N,C].")
        .Attr(
            "kernel_type",
            "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.",
            AttributeProto::STRING,
            std::string("LINEAR"))
        .Attr(
            "kernel_params",
            "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr("support_vectors", "Chosen support vectors", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "one_class",
            "Flag indicating whether the regression is a one-class SVM or not.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr("coefficients", "Support vector coefficients.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("n_supports", "The number of support vectors.", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr("rho", "", AttributeProto::FLOATS, OPTIONAL_VALUE));

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    5,
    OpSchema()
        .Deprecate()
        .SetDoc(kDoc_TreeEnsembleClassifier_ver5)
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
            OPTIONAL_VALUE));

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    5,
    OpSchema()
        .Deprecate()
        .SetDoc(kDoc_TreeEnsembleRegressor_ver5)
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
            OPTIONAL_VALUE));

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsemble,
    5,
    OpSchema()
        .SetDoc(kDoc_TreeEnsemble_ver5)
        .Input(0, "X", "Input of shape [Batch Size, Number of Features]", "T")
        .Output(0, "Y", "Output of shape [Batch Size, Number of targets]", "T")
        .TypeConstraint(
            "T",
            {types::Float, types::Double, types::Float16},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, true)
        .Attr(
            "nodes_splits",
            "Thresholds to do the splitting on for each node with mode that is not 'BRANCH_MEMBER'.",
            AttributeProto::TENSOR,
            true)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The comparison operation performed by the node. This is encoded as an enumeration of 0 ('BRANCH_LEQ'), 1 "
            "('BRANCH_LT'), 2 ('BRANCH_GTE'), 3 ('BRANCH_GT'), 4 ('BRANCH_EQ'), 5 ('BRANCH_NEQ'), and 6 "
            "('BRANCH_MEMBER'). Note this is a tensor of type uint8.",
            AttributeProto::TENSOR,
            true)
        .Attr(
            "nodes_truenodeids",
            "If `nodes_trueleafs` is false at an entry, this represents the position of the true branch node. This "
            "position can be used to index into a `nodes_*` entry. If `nodes_trueleafs` is false, it is an index into "
            "the leaf_* attributes.",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_falsenodeids",
            "If `nodes_falseleafs` is false at an entry, this represents the position of the false branch node. This "
            "position can be used to index into a `nodes_*` entry. If `nodes_falseleafs` is false, it is an index into "
            "the leaf_* attributes.",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_trueleafs",
            "1 if true branch is leaf for each node and 0 an interior node. To represent a tree that is a leaf (only "
            "has one node), one can do so by having a single `nodes_*` entry with true and false branches referencing "
            "the same `leaf_*` entry",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_falseleafs",
            "1 if false branch is leaf for each node and 0 if an interior node. To represent a tree that is a leaf "
            "(only has one node), one can do so by having a single `nodes_*` entry with true and false branches "
            "referencing the same `leaf_*` entry",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define whether to follow the true branch (if attribute value is 1) or false branch (if "
            "attribute value is 0) in the presence of a NaN input feature. This attribute may be left undefined and "
            "the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "tree_roots",
            "Index into `nodes_*` for the root of each tree. The tree structure is derived from the branching of each "
            "node.",
            AttributeProto::INTS,
            true)
        .Attr(
            "membership_values",
            "Members to test membership of for each set membership node. List all of the members to test again in the "
            "order that the 'BRANCH_MEMBER' mode appears in `node_modes`, delimited by `NaN`s. Will have the same "
            "number "
            "of sets of values as nodes with mode 'BRANCH_MEMBER'. This may be omitted if the node doesn't contain any "
            "'BRANCH_MEMBER' nodes.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "leaf_targetids",
            "The index of the target that this leaf contributes to (this must be in range `[0, n_targets)`).",
            AttributeProto::INTS,
            true)
        .Attr("leaf_weights", "The weight for each leaf.", AttributeProto::TENSOR, true)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE' (0), 'SOFTMAX' (1), 'LOGISTIC' (2), "
            "'SOFTMAX_ZERO' (3) or 'PROBIT' (4), defaults to 'NONE' (0)",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE' (0) 'SUM' (1) 'MIN' (2) 'MAX "
            "(3) defaults to 'SUM' (1)",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    OpSchema()
        .SetDoc(kDoc_ZipMap_ver1)
        .Input(0, "X", "The input values", "tensor(float)")
        .Output(0, "Z", "The output map", "T")
        .TypeConstraint(
            "T",
            {types::Sequence(types::Map<TensorProto::STRING>("float")),
             types::Sequence(types::Map<TensorProto::INT64>("float"))},
            "The output will be a sequence of string or integer maps to float.")
        .Attr(
            "classlabels_strings",
            "The keys when using string keys.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "The keys when using int keys.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
