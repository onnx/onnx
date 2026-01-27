// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "onnx/defs/controlflow/utils.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;

static std::vector<std::string> control_flow_types_ir13() {
  auto t = OpSchema::all_tensor_types_ir13();
  auto s = OpSchema::all_tensor_sequence_types_ir13();
  auto o = OpSchema::all_optional_types_ir13();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

ONNX_OPERATOR_SET_SCHEMA(
    If,
    25,
    OpSchema()
        .SetDoc("If conditional")
        .Input(0, "cond", "Condition for the if. The tensor must contain a single element.", "B")
        .Output(
            0,
            "outputs",
            "Values that are live-out to the enclosing scope. The return values in "
            "the `then_branch` and `else_branch` must be of the same data type. "
            "The `then_branch` and `else_branch` may produce tensors with the same "
            "element type and different shapes. "
            "If corresponding outputs from the then-branch and the else-branch have "
            "static shapes S1 and S2, then the shape of the corresponding output "
            "variable of the if-node (if present) must be compatible with both S1 "
            "and S2 as it represents the union of both possible shapes."
            "For example, if in a model file, the first "
            "output of `then_branch` is typed float tensor with shape [2] and the "
            "first output of `else_branch` is another float tensor with shape [3], "
            "If's first output should have (a) no shape set, or (b) "
            "a shape of rank 1 with neither `dim_value` nor `dim_param` set, or (c) "
            "a shape of rank 1 with a unique `dim_param`. "
            "In contrast, the first output cannot have the shape [2] since [2] and "
            "[3] are not compatible.",
            "V",
            OpSchema::Variadic,
            false)
        .Attr(
            "then_branch",
            "Graph to run if condition is true. Has N outputs: values you wish to "
            "be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the else_branch.",
            AttributeProto::GRAPH)
        .Attr(
            "else_branch",
            "Graph to run if condition is false. Has N outputs: values you wish to"
            " be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the then_branch.",
            AttributeProto::GRAPH)
        .TypeConstraint(
            "V",
            control_flow_types_ir13(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv13.")
        .TypeConstraint("B", {"tensor(bool)"}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction));

static const char* const Loop_ver25_doc = kDoc_Loop_ver23;

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    25,
    OpSchema()
        .SetDoc(Loop_ver25_doc)
        .Input(
            0,
            "M",
            "A maximum trip-count for the loop specified at runtime. Optional."
            " Pass empty string to skip.",
            "I",
            OpSchema::Optional)
        .Input(
            1,
            "cond",
            "A boolean termination condition. Optional. Pass empty string to skip.",
            "B",
            OpSchema::Optional)
        .Input(
            2,
            "v_initial",
            "The initial values of any loop-carried dependencies (values that "
            "change across loop iterations)",
            "V",
            OpSchema::Variadic,
            false,
            0)
        .Output(
            0,
            "v_final_and_scan_outputs",
            "Final N loop carried dependency values then K scan_outputs. "
            "Scan outputs must be Tensors.",
            "V",
            OpSchema::Variadic,
            false)
        .Attr(
            "body",
            "The graph run each iteration. It has 2+N inputs: (iteration_num, "
            "condition, loop carried dependencies...). It has 1+N+K outputs: "
            "(condition, loop carried dependencies..., scan_outputs...). Each "
            "scan_output is created by concatenating the value of the specified "
            "output value at the end of each iteration of the loop. It is an error"
            " if the dimensions or data type of these scan_outputs change across loop"
            " iterations.",
            AttributeProto::GRAPH)
        .TypeConstraint(
            "V",
            control_flow_types_ir13(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv13.")
        .TypeConstraint("I", {"tensor(int64)"}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {"tensor(bool)"}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));

static const char* const scan_25_doc = kDoc_scan_24;

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    25,
    OpSchema()
        .SetDoc(scan_25_doc)
        .Input(
            0,
            "initial_state_and_scan_inputs",
            "Initial values of the loop's N state variables followed by M scan_inputs",
            "V",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "final_state_and_scan_outputs",
            "Final values of the loop's N state variables followed by K scan_outputs",
            "V",
            OpSchema::Variadic,
            false)
        .Attr(
            "body",
            "The graph run each iteration. It has N+M inputs: "
            "(loop state variables..., scan_input_elts...). It has N+K outputs: "
            "(loop state variables..., scan_output_elts...). Each "
            "scan_output is created by concatenating the value of the specified "
            "scan_output_elt value at the end of each iteration of the loop. It is an error"
            " if the dimensions of these values change across loop iterations.",
            AttributeProto::GRAPH,
            true)
        .Attr("num_scan_inputs", "An attribute specifying the number of scan_inputs M. ", AttributeProto::INT, true)
        .Attr(
            "scan_input_directions",
            "An optional list of M flags. The i-th element of the list specifies the direction "
            "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
            "indicates reverse direction. "
            "If omitted, all scan_input tensors will be scanned in the forward direction.",
            AttributeProto::INTS,
            false)
        .Attr(
            "scan_output_directions",
            "An optional list of K flags, one for each scan_output. The i-th element of the list "
            "specifies whether the i-th scan_output should be constructed by appending or "
            "prepending a new value in each iteration: 0 indicates appending and 1 "
            "indicates prepending. "
            "If omitted, all scan_output tensors will be produced by appending a value "
            "in each iteration.",
            AttributeProto::INTS,
            false)
        .Attr(
            "scan_input_axes",
            "An optional list of M flags. The i-th element of the list specifies the axis "
            "to be scanned (the sequence axis) for the i-th scan_input. If omitted, 0 will "
            "be used as the scan axis for every scan_input. Negative value for an axis means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INTS,
            false)
        .Attr(
            "scan_output_axes",
            "An optional list of K flags. The i-th element of the list specifies the axis "
            "for the i-th scan_output. The scan outputs are accumulated along the specified "
            "axis. If omitted, 0 will be used as the scan axis for every scan_output. "
            "Negative value for an axis means counting dimensions from the back. Accepted "
            "range is [-r, r-1].",
            AttributeProto::INTS,
            false)
        .TypeConstraint("V", OpSchema::all_tensor_types_ir13(), "All Tensor types up to IRv13.")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

} // namespace ONNX_NAMESPACE
