// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
const char* reduction_doc =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': the output is the loss for each sample in the batch."
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the batch_size.";

static const char* MSE_ver12_doc = R"DOC(Loss function that measures the
mean squared error (squared L2 norm) between each element in the 'scores'
and 'labels'.

The loss can be described as:
    L = (l_1, l_2, ..., l_N), l_n = (score_n - label_n)^2
, where N is the batch size.

If 'weights' is provided, it should be broadcastable to shape of 'scores'.
    L = Mul(weights, L)
, where Mul is element-wise binary multiplication with Numpy-style broadcasting support.

Finally, L is reduced:
L = ReduceSum(L), if reduction = 'sum';
    ReduceMean(L), if reduction = 'mean';
    ReduceMean(L, axes=[0]), if reduction = 'none';

.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MeanSquaredError,
    12,
    OpSchema()
        .SetDoc(MSE_ver10_doc)
        .Attr(
            "reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
        .Input(0, "scores", "The predicted outputs.", "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, same dimensions as 'scores'.",
            "T")
        .Input(
            2,
            "weights",
            "Weights acts as a coefficient for the loss, it should be "
            "broadcastable to shape of 'scores'.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "output",
            "Weighted loss float Tensor. If reduction is none, this has the "
            "shape of [batch_size]; otherwise, it is scalar.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {}));

} // namespace ONNX_NAMESPACE

