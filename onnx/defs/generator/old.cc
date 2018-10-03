// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Constant_ver1_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    OpSchema()
        .SetDoc(Constant_ver1_doc)
        .Attr(
            "value",
            "The value for the elements of the output tensor.",
            AttributeProto::TENSOR)
        .Output(
            0,
            "output",
            "Output tensor containing the same value of the provided tensor.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto)
            return; // attribute not present
          if (!attr_proto->has_t())
            return; // attribute has no tensor value
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));
} // namespace ONNX_NAMESPACE
