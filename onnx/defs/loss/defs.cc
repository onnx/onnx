// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
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

scores: (N, C) where C is the number of classes, or (N, C, d1, d2,..., dk),
	with K >= 1 in case of K-dimensional loss.
labels: (N) where each value is 0 <= labels[i] <= C-1, or (N, d1, d2,..., dk),
	with K >= 1 in case of K-dimensional loss.

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
	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "none"
              return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "none"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
		{{"X_SM"}, "Softmax", {"scores"}},
		{{"X_Log"}, "Log", {"X_SM"}},
                {{"output"}, "Mul", {"labels", "X_Log"}}
		}))
	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "sum"
              return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "sum"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                {{"X_SM"}, "Softmax", {"scores"}},
                {{"X_Log"}, "Log", {"X_SM"}},
                {{"X_Mul"}, "Mul", {"labels", "X_Log"}},
                {{"output"}, "ReduceSum", {"X_Mul"}}
                }))
	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "mean"
              return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "mean"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                {{"X_SM"}, "Softmax", {"scores"}},
                {{"X_Log"}, "Log", {"X_SM"}},
                {{"X_Mul"}, "Mul", {"labels", "X_Log"}},
                {{"output"}, "ReduceMean", {"X_Mul"}}
                }))
	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // weight, reduction is "none"
              return ctx.getNumInputs() > 2 && ctx.getAttribute("reduction")->s() == "none"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                {{"X_SM"}, "Softmax", {"scores"}},
                {{"X_Log"}, "Log", {"X_SM"}},
                {{"X_Mul"}, "Mul", {"labels", "X_Log"}},
                {{"output"}, "Mul", {"weights", "X_Mul"}}
                }))
        .AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // weight, reduction is "sum"
              return ctx.getNumInputs() > 2 && ctx.getAttribute("reduction")->s() == "sum"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                {{"X_SM"}, "Softmax", {"scores"}},
                {{"X_Log"}, "Log", {"X_SM"}},
                {{"X_Mul"}, "Mul", {"labels", "X_Log"}},
		{"X_Mul2"}, "Mul", {"weights", "X_Mul"}},
                {{"output"}, "ReduceSum", {"X_Mul2"}}
                }))
        .AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "mean"
              return ctx.getNumInputs() > 2 && ctx.getAttribute("reduction")->s() == "mean"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                {{"X_SM"}, "Softmax", {"scores"}},
                {{"X_Log"}, "Log", {"X_SM"}},
                {{"X_Mul"}, "Mul", {"labels", "X_Log"}},
		{{"X_Mul2"}, "Mul", {"weights", "X_Mul"}},
                {{"output"}, "ReduceMean", {"X_Mul2"}}
                }))
	.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
	    propagateElemTypeFromInputToOutput(ctx, 0, 0);
	    auto& scores_input_shape = getInputShape(ctx, 0);
	    auto& labels_input_shape = getInputShape(ctx, 1);
	    if (scores_input_shape.dim_size() != labels_input_shape.dim_size()) {
		fail_shape_inference("scores and labels must be of the same rank.");
	    }
	    std::string reduction = getAttribute(ctx, "reduction", "mean");
	    if (reduction.compare("none") == 0 && hasInputShape(ctx, 0)) {
		propagateShapeFromInputToOutput(ctx, 0, 0);
	    }

	}));

} // namespace ONNX_NAMESPACE
