// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
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

bool BuildContextDependentFunctionBodyMSD(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back(FunctionBodyHelper::Const<int>("Q_Pow", 2));
  body.push_back({{"X_Sub"}, "Sub", {"scores", "labels"}});

  if (ctx.hasInput(2)) {
    if (ctx.getAttribute("reduction")->s() == "none") {
      body.push_back({{"output"}, "Pow", {"X_Sub", "Q_Pow"}});
    } else {
      body.push_back({{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}});
      if (ctx.getAttribute("reduction")->s() == "mean") {
        body.push_back({{"output"}, "ReduceMean", {"X_Pow"}});
      } else {
        body.push_back({{"output"}, "ReduceSum", {"X_Pow"}});
      }
    }
  } else {
    body.push_back({{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}});
    if (ctx.getAttribute("reduction")->s() == "none") {
      body.push_back({{"output"}, "Mul", {"weights", "X_Pow"}});
    } else {
      body.push_back({{"X_Mul"}, "Mul", {"weights", "X_Pow"}});
      if (ctx.getAttribute("reduction")->s() == "mean") {
        body.push_back({{"output"}, "ReduceMean", {"X_Mul"}});
      } else {
        body.push_back({{"output"}, "ReduceSum", {"X_Mul"}});
      }
    }
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
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyMSD)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
            std::string reduction = getAttribute(ctx, "reduction", "mean");
            if (reduction.compare("none") == 0 && hasInputShape(ctx, 0)) {
                propagateShapeFromInputToOutput(ctx, 0, 0);
            }
        }));

} // namespace ONNX_NAMESPACE

