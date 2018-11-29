// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {
static const char* LabelEncoder_ver1_doc = R"DOC(
    Converts strings to integers and vice versa.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.<br>
    Each operator converts either integers to strings or strings to integers, depending 
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    When converting from integers to strings, the string is fetched from the
    'classes_strings' list, by simple indexing.<br>
    When converting from strings to integers, the string is looked up in the list
    and the index at which it is found is used as the converted value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    1,
    OpSchema()
        .SetDoc(LabelEncoder_ver1_doc)
        .Input(0, "X", "Input data.", "T1")
        .Output(0, "Y", "Output data. If strings are input, the output values are integers, and vice versa.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)"},
            "The input type must be a tensor of integers or strings, of any shape.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, and will have the same shape as the input.")
        .Attr(
            "classes_strings",
            "A list of labels.",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "default_int64",
            "An integer to use when an input string value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "default_string",
            "A string to use when an input integer value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
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
} // namespace ONNX_NAMESPACE
#endif
