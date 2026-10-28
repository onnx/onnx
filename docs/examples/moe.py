# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, parser


def make_functions() -> list[onnx.FunctionProto]:
    """Build portable model-local functions used by the MoE examples."""
    bodies = [
        """
        MixingWeights (Scores, Indices, Normalize, Epsilon, Scale) => (Weights) {
            Selected = GatherElements <axis = 1> (Scores, Indices)
            Axis = Constant <value = int64[1] {1}> ()
            Sum = ReduceSum <keepdims = 1> (Selected, Axis)
            Stabilized = Add (Sum, Epsilon)
            One = Constant <value = float {1}> ()
            Denominator = Where (Normalize, Stabilized, One)
            Normalized = Div (Selected, Denominator)
            Weights = Mul (Normalized, Scale)
        }
        """,
        """
        SoftmaxRouter (Logits, K, Normalize, Epsilon, Scale) => (Indices, Weights) {
            Scores = Softmax <axis = 1> (Logits)
            Values, Indices = TopK <axis = 1, largest = 1, sorted = 1> (Scores, K)
            Weights = local.moe.MixingWeights (Scores, Indices, Normalize, Epsilon, Scale)
        }
        """,
        """
        GroupedRouter (Logits, Correction, GroupShape, GroupK, K, Normalize,
                       Epsilon, Scale) => (Indices, Weights) {
            Scores = Sigmoid (Logits)
            Choice = Add (Scores, Correction)
            TokenShape = Shape <start = 0, end = 1> (Logits)
            GroupedShape = Concat <axis = 0> (TokenShape, GroupShape)
            Grouped = Reshape <allowzero = 1> (Choice, GroupedShape)
            Two = Constant <value = int64[1] {2}> ()
            BestTwo, BestTwoIndices = TopK <axis = 2> (Grouped, Two)
            LastAxis = Constant <value = int64[1] {2}> ()
            GroupScores = ReduceSum <keepdims = 0> (BestTwo, LastAxis)
            GroupValues, Groups = TopK <axis = 1, sorted = 1> (GroupScores, GroupK)
            MaskShape = Shape (GroupScores)
            EmptyMask = ConstantOfShape <value = bool[1] {0}> (MaskShape)
            UpdateShape = Shape (Groups)
            Updates = ConstantOfShape <value = bool[1] {1}> (UpdateShape)
            GroupMask = ScatterElements <axis = 1> (EmptyMask, Groups, Updates)
            UnsqueezedMask = Unsqueeze (GroupMask, LastAxis)
            ExpandedMask = Expand (UnsqueezedMask, GroupedShape)
            ScoreShape = Shape (Scores)
            Mask = Reshape <allowzero = 1> (ExpandedMask, ScoreShape)
            NegativeInfinity = Constant <value = float {-inf}> ()
            Eligible = Where (Mask, Choice, NegativeInfinity)
            Values, Indices = TopK <axis = 1, sorted = 1> (Eligible, K)
            Weights = local.moe.MixingWeights (Scores, Indices, Normalize, Epsilon, Scale)
        }
        """,
        """
        SelectedSwiGLU (X, Gate, Up, Down, Indices, Weights) => (Y) {
            GateValues = GroupedMatMul (X, Gate, Indices)
            UpValues = GroupedMatMul (X, Up, Indices)
            SigmoidGate = Sigmoid (GateValues)
            ActivatedGate = Mul (GateValues, SigmoidGate)
            Hidden = Mul (ActivatedGate, UpValues)
            RouteShape = Shape (Indices)
            FlatRows = ReduceProd <keepdims = 1> (RouteShape)
            IntermediateShape = Shape <start = 2, end = 3> (Hidden)
            FlatHiddenShape = Concat <axis = 0> (FlatRows, IntermediateShape)
            FlatHidden = Reshape <allowzero = 1> (Hidden, FlatHiddenShape)
            FlatIndexShape = Constant <value = int64[2] {-1, 1}> ()
            FlatIndices = Reshape (Indices, FlatIndexShape)
            Projected = GroupedMatMul (FlatHidden, Down, FlatIndices)
            HiddenShape = Shape <start = 1, end = 2> (X)
            OutputShape = Concat <axis = 0> (RouteShape, HiddenShape)
            ExpertOutputs = Reshape <allowzero = 1> (Projected, OutputShape)
            LastAxis = Constant <value = int64[1] {2}> ()
            ExpandedWeights = Unsqueeze (Weights, LastAxis)
            Weighted = Mul (ExpertOutputs, ExpandedWeights)
            RouteAxis = Constant <value = int64[1] {1}> ()
            Y = ReduceSum <keepdims = 0> (Weighted, RouteAxis)
        }
        """,
    ]
    functions = [
        parser.parse_function(
            '<domain: "local.moe", opset_import: ["" : 29, "local.moe" : 1]>\n' + body
        )
        for body in bodies
    ]
    selected = functions[-1]
    # These types let the evaluator expand GroupedMatMul's context-dependent body.
    float_shapes = {
        "X": ["T", "H"],
        "Gate": ["E", "H", "I"],
        "Up": ["E", "H", "I"],
        "Down": ["E", "I", "H"],
        "Weights": ["T", "k"],
        "FlatHidden": ["Tk", "I"],
    }
    selected.value_info.extend(
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in float_shapes.items()
    )
    selected.value_info.extend(
        [
            helper.make_tensor_value_info(
                "Indices", onnx.TensorProto.INT64, ["T", "k"]
            ),
            helper.make_tensor_value_info(
                "FlatIndices", onnx.TensorProto.INT64, ["Tk", 1]
            ),
        ]
    )
    return functions


def make_model(*, grouped: bool = False) -> onnx.ModelProto:
    """Build a float32 MoE block with dynamic token/expert/hidden dimensions.

    The softmax example has a sigmoid-gated shared expert (Qwen family).
    The grouped sigmoid example has an ungated shared expert (GLM/Kimi family).
    See the accompanying technical note for the input contract.
    """
    router = (
        """Indices, Weights = local.moe.GroupedRouter (
               Logits, Correction, GroupShape, GroupK, K, Normalize, Epsilon, Scale)"""
        if grouped
        else """Indices, Weights = local.moe.SoftmaxRouter (
                    Logits, K, Normalize, Epsilon, Scale)"""
    )
    shared = (
        "Shared = Identity (SharedOutput)"
        if grouped
        else """
            SharedLogit = MatMul (X, SharedRouter)
            SharedWeight = Sigmoid (SharedLogit)
            Shared = Mul (SharedOutput, SharedWeight)
        """
    )
    extra_inputs = (
        "float[E] Correction, int64[2] GroupShape, int64[1] GroupK,"
        if grouped
        else "float[H,1] SharedRouter,"
    )
    graph = parser.parse_graph(
        f"""
        moe (
            float[T,H] X, float[H,E] Router,
            float[E,H,I] Gate, float[E,H,I] Up, float[E,I,H] Down,
            float[H,J] SharedGate, float[H,J] SharedUp, float[J,H] SharedDown,
            {extra_inputs}
            int64[1] K, bool Normalize, float Epsilon, float Scale
        ) => (float[T,H] Y, int64[T,k] Indices, float[T,k] Weights) {{
            Logits = MatMul (X, Router)
            {router}
            Routed = local.moe.SelectedSwiGLU (X, Gate, Up, Down, Indices, Weights)
            SharedGateValues = MatMul (X, SharedGate)
            SharedUpValues = MatMul (X, SharedUp)
            SharedSigmoid = Sigmoid (SharedGateValues)
            SharedActivated = Mul (SharedGateValues, SharedSigmoid)
            SharedHidden = Mul (SharedActivated, SharedUpValues)
            SharedOutput = MatMul (SharedHidden, SharedDown)
            {shared}
            Y = Add (Routed, Shared)
        }}
        """
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 29),
            helper.make_opsetid("local.moe", 1),
        ],
        functions=make_functions(),
        producer_name="onnx-moe-example",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Write portable opset-29 MoE models.")
    args.add_argument("output", type=Path, help="Directory for the generated models")
    output = args.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), output / "softmax_moe.onnx")
    onnx.save(make_model(grouped=True), output / "grouped_sigmoid_moe.onnx")
