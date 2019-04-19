// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
const char* reduction_doc =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': no reduction will be applied, "
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the number of "
    "elements in the output.";

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
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
        .Input(0, "predictions", "The predicted outputs.", "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, same dimensions as 'predictions'.",
            "T")
        .Input(
            2,
            "weights",
            "Weights acts as a coefficient for the loss. If a scalar is provided, "
            "then the loss is simply scaled by the given value. If weights is a "
            "tensor of size [batch_size], then the total loss for each sample of the "
            "batch is rescaled by the corresponding element in the weights vector. "
            "If the shape of weights matches the shape of predictions, then the loss "
            "of each measurable element of predictions is scaled by the corresponding "
            "value of weights.",
            "T",
            OpSchema::Optional)
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
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {}));

static const char* SoftmaxCrossEntropy_ver10_doc =
    R"DOC(Loss function that measures the softmax cross entropy between 
each element in the predictions and labels.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropy,
    10,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropy_ver10_doc)
        .Attr(
            "reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
        .Input(
            0,
            "predictions",
            "The predicted outputs with shape [batch_size, class_size], or "
            "[batch_size, class_size, d1, d2 , ..., dk], where K is the number of dimensions.",
            "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, same dimensions as 'predictions'. "
            "Usualy, it's a one-hot representation of groud-truth class.",
            "T")
        .Input(
            2,
            "weights",
            "A manual rescaling weight given to each class. If given, it has to "
            "be a 1D Tensor assigning weight to each of the classes. Otherwise, "
            "it is treated as if having all ones.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "output",
            "Weighted loss float Tensor. If reduction is 'none', this has the "
            "shape of [batch_size], or [batch_size, d1, d2, ..., dk] in case of "
            "K-dimensional loss. Otherwise, it is a scalar.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {}));

} // namespace ONNX_NAMESPACE
