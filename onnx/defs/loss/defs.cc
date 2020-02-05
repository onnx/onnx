// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
const char* reduction_doc =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': the output is the loss for each sample in the batch."
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the batch_size.";

static const char* MSD_ver12_doc = R"DOC(Loss function that measures the
mean squared distance (squared L2 norm) between each element in the 'scores'
and 'labels'.

The loss can be described as:
    L = (l_1, l_2, ..., l_N), l_n = (score_n - label_n)^2
, N is the batch size.

score and label are vectors of arbitrary shapes with total of N elements each.

If 'weights' is provided, it should be broadcastable to shape of 'scores'.
    L = Mul(weights, L)
, where Mul is element-wise binary multiplication with Numpy-style broadcasting support.

Finally, L is reduced:
L = ReduceSum(L), if reduction = 'sum';
    ReduceMean(L), if reduction = 'mean';
    L, if reduction = 'none';

.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MeanSquaredDistance,
    12,
    OpSchema()
        .SetDoc(MSD_ver12_doc)
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
	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "none"
	      return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "none"; },
	    FunctionBodyHelper::BuildNodes({
	        // nodes: {outputs, op, inputs, attributes}
	        FunctionBodyHelper::Const<int>("Q_Pow", 2), 
	        {{"X_Sub"}, "Sub", {"scores", "labels"}},
                {{"output"}, "Pow", {"X_Sub", "Q_Pow"}}
	        }))

	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "mean"
              return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "mean"; }, 
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                FunctionBodyHelper::Const<int>("Q_Pow", 2),
                {{"X_Sub"}, "Sub", {"scores", "labels"}},
                {{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}},
		{{"output"}, "ReduceMean", {"X_Pow"}}
                }))

	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // no weight, reduction is "sum"
              return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "sum"; }, 
            FunctionBodyHelper::BuildNodes({ 
                // nodes: {outputs, op, inputs, attributes}
                FunctionBodyHelper::Const<int>("Q_Pow", 2),
                {{"X_Sub"}, "Sub", {"scores", "labels"}}, 
                {{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}},
                {{"output"}, "ReduceSum", {"X_Pow"}}
                }))

	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // weight, reduction is "none"
              return ctx.getNumInputs() == 2 && ctx.getAttribute("reduction")->s() == "none"; },
	    FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
		FunctionBodyHelper::Const<int>("Q_Pow", 2),                                                                                                           
		{{"X_Sub"}, "Sub", {"scores", "labels"}},
		{{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}},
                {{"output"}, "Mul", {"weights", "X_Pow"}}
                }))

	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // weight, reduction is "mean"
              return ctx.getNumInputs() > 2 && ctx.getAttribute("reduction")->s() == "mean"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                FunctionBodyHelper::Const<int>("Q_Pow", 2),
                {{"X_Sub"}, "Sub", {"scores", "labels"}},
                {{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}},
                {{"X_Mul"}, "Mul", {"weights", "X_Pow"}},
		{{"output"}, "ReduceMean", {"X_Mul"}}
                }))

	.AddQueriedFunctionBody([](FunctionBodyQueryContext& ctx) { // weight, reduction is "sum"
              return ctx.getNumInputs() > 2 && ctx.getAttribute("reduction")->s() == "sum"; },
            FunctionBodyHelper::BuildNodes({
                // nodes: {outputs, op, inputs, attributes}
                FunctionBodyHelper::Const<int>("Q_Pow", 2),
                {{"X_Sub"}, "Sub", {"scores", "labels"}},
                {{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}},
                {{"X_Mul"}, "Mul", {"weights", "X_Pow"}},
		{{"output"}, "ReduceSum", {"X_Mul"}}
                }))
	.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
	    propagateElemTypeFromInputToOutput(ctx, 0, 0);
	    std::string reduction = getAttribute(ctx, "reduction", "mean");
	    if (reduction.compare("none") == 0 && hasInputShape(ctx, 0)) {
	    	propagateShapeFromInputToOutput(ctx, 0, 0);
	    }
	}));

} // namespace ONNX_NAMESPACE

