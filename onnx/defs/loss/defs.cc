// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
const char* reduction_doc =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': the output is the loss for each sample in the batch."
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the batch_size.";

static const char* MSE_ver10_doc = R"DOC(Loss function that measures the
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
    10,
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

static const char* SoftmaxCrossEntropy_ver10_doc =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.

The loss can be described as:
    L = (l_1, l_2, ..., l_N), where N is the batch_size

The loss for one sample, l_n, can caculated as follows
    let p = Softmax(scores)
    l_n = -sum(label_i * log(p_i)), where i is the index of classes.
or
    l_n = -sum(weight_i * label_i * log(p_i)), if 'weights' is provided.

Finally, L is reduced:
L = ReduceSum(L), if reduction = 'sum';
    ReduceMean(L), if reduction = 'mean';
    L, if reduction = 'none'

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropy,
    10,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropy_ver10_doc)
        .Attr(
            " ",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
        .Input(
            0,
            "scores",
            "The predicted outputs with shape [batch_size, class_size], or "
            "[batch_size, class_size, d1, d2 , ..., dk], where K is the number of dimensions.",
            "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, same dimensions as 'scores'. "
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
