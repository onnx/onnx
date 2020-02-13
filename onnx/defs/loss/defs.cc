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
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, d1, d2, ..., dk),
the loss tensor L may have (N, d1, d2, ..., dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

where:
    p = Softmax(scores)
    y = log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
L = ReduceSum(L), if reduction = 'sum';
    ReduceMean(L), if reduction = 'mean'; if "weight" is provided, output is averaged by sum of weights.
    L, if reduction = 'none'
)DOC";

bool BuildContextDependentFunctionBodySCE(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back({{"X_Max"}, "Max", {"scores"}});
  body.push_back({{"X_Sub"}, "Sub", {"scores", "X_Max"}});
  body.push_back({{"X_Exp"}, "Exp", {"X_Sub"}});
  body.push_back({{"X_RS"}, "ReduceSum", {"X_Exp"}});
  body.push_back({{"X_Div"}, "Div", {"X_Exp", "X_RS"}});
  body.push_back({{"X_Log"}, "Log", {"X_Div"}});
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
            "[batch_size, class_size, D1, D2 , ..., Dk], where K is the number of dimensions.",
            "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, with shape [batch_size], or "
            "[batch_size, D1, D2, ..., Dk], where K is the number of dimensions."
            "Usualy, it's a one-hot representation of ground-truth class.",
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
            "shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of "
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
            if (reduction.compare("none") == 0) {
	        if (hasInputShape(ctx, 1)) {
                    propagateShapeFromInputToOutput(ctx, 1, 0);
		}
            } else {
	        updateOutputShape(ctx, 0, TensorShapeProto());
	    }

        }));

} //namespace ONNX_NAMESPACE
