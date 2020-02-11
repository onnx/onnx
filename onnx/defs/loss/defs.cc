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

static const char* SoftmaxCrossEntropyLoss_ver12_doc =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
The loss can be described as:
    L = (l_1, l_2, ..., l_N), where N is the batch_size

shape(scores): (N, C) where C is the number of classes, or (N, C, d1, d2,..., dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, d1, d2,..., dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l_i = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l_i = -y[i][c][d1][d2]..[dk]*weights[c], if 'weights' is provided.

where:
    p = Softmax(scores)
    y = log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is reduced:
L = ReduceSum(L), if reduction = 'sum';
    ReduceMean(L), if reduction = 'mean'; if "weight" is provided, output is averaged by sum of weights.
    L, if reduction = 'none'
)DOC";

bool BuildContextDependentFunctionBodySCE(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back({{"X_SM"}, "Softmax", {"scores"}});
  body.push_back({{"X_Log"}, "Log", {"X_SM"}});
  if (ctx.hasInput(2)) {
    body.push_back({{"output"}, "NegativeLogLikelihoodLoss", {"X_Log", "labels"}, {MakeAttribute("reduction", "mean")}});
  } else {
    body.push_back({{"output"}, "NegativeLogLikelihoodLoss", {"X_Log", "labels", "weights"}, {MakeAttribute("reduction", "mean")}});
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    12,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropyLoss_ver12_doc)
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
            "The ground truth output tensor, with shape [batch_size], or "
            "[batch_size, d1, d2 , ..., dk], where K is the number of dimensions."
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
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySCE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
            std::string reduction = getAttribute(ctx, "reduction", "mean");
            if (reduction.compare("none") == 0 && hasInputShape(ctx, 1)) {
                propagateShapeFromInputToOutput(ctx, 1, 0);
            }

        }));

} // namespace ONNX_NAMESPACE
