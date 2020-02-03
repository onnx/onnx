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

static const char* SoftmaxCrossEntropy_ver12_doc =
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
    12,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropy_ver12_doc)
        .Attr(
            "reduction",
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
	.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
	    propagateElemTypeFromInputToOutput(ctx, 0, 0);
	    std::string reduction = getAttribute(ctx, "reduction", "mean");
	    if (reduction.compare("none") == 0) {
		propagateShapeFromInputToOutput(ctx, 0, 0);
	    }

	}));

} // namespace ONNX_NAMESPACE
