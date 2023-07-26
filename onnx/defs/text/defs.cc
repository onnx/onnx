/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* ParseDateTime_doc = R"DOC(Parse a datetime string into a (floating point) Unix time stamp.)DOC";
ONNX_OPERATOR_SET_SCHEMA(
    ParseDateTime,
    20,
    OpSchema()
        .Input(0, "X", "Tensor with datetime strings", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "y", "Unix time stamps", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr("format", "Format description in the syntax of C's `strptime`.", AttributeProto::STRING)
        .Attr(
            "unit",
            "Unit of the returned time stamp. Allowed values are: 's' (second), 'ms' (millisecond), 'us' (microsecond) or 'ns' (nanosecond).",
            AttributeProto::STRING)
        .Attr(
            "default",
            "Default value to be used if the parsing fails. The tensor must be of rank 0 and either of type `tensor(int64)` or `tensor(double)`. The tensor type is the output type. If 'default' is specified, the output type is `tensor(int64)` and the behavior for failing to parse an input element is implementation defined.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)

        .TypeConstraint("T1", {"tensor(string)"}, "UTF-8 datetime strings")
        .TypeConstraint("T2", {"tensor(double)", "tensor(int64)"}, "Output type depends on 'default' attribute.")
        .SetDoc(ParseDateTime_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* default_value = ctx.getAttribute("default");

          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }

          if (nullptr == default_value) {
            updateOutputElemType(ctx, 0, TensorProto::INT64);
            return;
          } else {
            const TensorProto& tensor_proto = default_value->t();
            updateOutputElemType(ctx, 0, tensor_proto.data_type());
            return;
          }
        }));

static const char* StringConcat_doc =
    R"DOC(StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support))DOC";
ONNX_OPERATOR_SET_SCHEMA(
    StringConcat,
    20,
    OpSchema()
        .Input(
            0,
            "X",
            "Tensor to prepend in concatenation",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(1, "Y", "Tensor to append in concatenation", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Z", "Concatenated string tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", {"tensor(string)"}, "Inputs and outputs must be UTF-8 strings")
        .SetDoc(StringConcat_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* StringNormalizer_ver10_doc = R"DOC(
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringNormalizer,
    10,
    OpSchema()
        .Input(0, "X", "UTF-8 strings to normalize", "tensor(string)")
        .Output(0, "Y", "UTF-8 Normalized strings", "tensor(string)")
        .Attr(
            std::string("case_change_action"),
            std::string("string enum that cases output to be lowercased/uppercases/unchanged."
                        " Valid values are \"LOWER\", \"UPPER\", \"NONE\". Default is \"NONE\""),
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            std::string("is_case_sensitive"),
            std::string("Boolean. Whether the identification of stop words in X is case-sensitive. Default is false"),
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "stopwords",
            "List of stop words. If not set, no word would be removed from X.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "locale",
            "Environment dependent string that denotes the locale according to which output strings needs to be upper/lowercased."
            "Default en_US or platform specific equivalent as decided by the implementation.",
            AttributeProto::STRING,
            OPTIONAL_VALUE)
        .SetDoc(StringNormalizer_ver10_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::STRING);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          TensorShapeProto output_shape;
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto dim_size = input_shape.dim_size();
          // Last axis dimension is unknown if we have stop-words since we do
          // not know how many stop-words are dropped
          if (dim_size == 1) {
            // Unknown output dimension
            output_shape.add_dim();
          } else if (dim_size == 2) {
            // Copy B-dim
            auto& b_dim = input_shape.dim(0);
            if (!b_dim.has_dim_value() || b_dim.dim_value() != 1) {
              fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
            }
            *output_shape.add_dim() = b_dim;
            output_shape.add_dim();
          } else {
            fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
          }
          updateOutputShape(ctx, 0, output_shape);
        }));
} // namespace ONNX_NAMESPACE
