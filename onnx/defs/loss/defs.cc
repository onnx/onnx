// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {

static const char* MSE_ver10_doc = R"DOC(Loss function that measures the
mean squared error (squared L2 norm) between each element in the predictions
and labels.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MeanSquaredError,
    10,
    OpSchema()
        .SetDoc(MSE_ver10_doc)
        .Attr(
            "reduction",
            "Type of reduction to apply to loss: none, sum, mean(default). "
            "'none': no reduction will be applied, "
            "'sum': the output will be summed. "
            "'mean': the sum of the output will be divided by the number of "
            "elements in the output, ",
            AttributeProto::STRING,
            std::string("mean"))
        .Input(
          0,
          "predictions",
          "The predicted outputs.",
          "T")
        .Input(
          1,
          "labels",
          "The ground truth output tensor, same dimensions as 'predictions'.",
          "T")
        .Output(
            0,
            "output",
            "Weighted loss float Tensor. If reduction is none, this has the "
            "same shape as labels; otherwise, it is scalar.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx){}));

} // namespace ONNX_NAMESPACE
