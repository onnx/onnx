// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/controlflow/utils.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

using SupportType = OpSchema::SupportType;

static std::vector<std::string> control_flow_types_ir12() {
  auto t = OpSchema::all_tensor_types_ir12();
  auto s = OpSchema::all_tensor_sequence_types_ir12();
  auto o = OpSchema::all_optional_types_ir12();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

static std::vector<std::string> control_flow_types_ir11() {
  auto t = OpSchema::all_tensor_types_ir11();
  auto s = OpSchema::all_tensor_sequence_types_ir11();
  auto o = OpSchema::all_optional_types_ir11();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

static std::vector<std::string> control_flow_types_ir10() {
  auto t = OpSchema::all_tensor_types_ir10();
  auto s = OpSchema::all_tensor_sequence_types_ir10();
  auto o = OpSchema::all_optional_types_ir10();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

static std::vector<std::string> control_flow_types_ir9() {
  auto t = OpSchema::all_tensor_types_ir9();
  auto s = OpSchema::all_tensor_sequence_types_ir9();
  auto o = OpSchema::all_optional_types_ir9();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

static std::vector<std::string> control_flow_types_ir4() {
  auto t = OpSchema::all_tensor_types_ir4();
  auto s = OpSchema::all_tensor_sequence_types_ir4();
  auto o = OpSchema::all_optional_types_ir4();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

ONNX_OPERATOR_SET_SCHEMA(
    If,
    24,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
            control_flow_types_ir12(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv11.")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    23,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
            control_flow_types_ir11(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv11.")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    21,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
            control_flow_types_ir10(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv10.")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    19,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
            control_flow_types_ir9(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv9.")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    16,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
            control_flow_types_ir4(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv4.")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    24,
    OpSchema()
        .SetDoc(kDoc_Loop_ver23)
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
            control_flow_types_ir12(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv11.")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    23,
    OpSchema()
        .SetDoc(kDoc_Loop_ver23)
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
            control_flow_types_ir11(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv11.")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    21,
    OpSchema()
        .SetDoc(kDoc_Loop_ver23)
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
            control_flow_types_ir10(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv10.")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    19,
    OpSchema()
        .SetDoc(kDoc_Loop_ver23)
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
            control_flow_types_ir9(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv9.")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    16,
    OpSchema()
        .SetDoc(kDoc_Loop_ver23)
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
            control_flow_types_ir4(),
            "All Tensor, Sequence(Tensor), Optional(Tensor), and Optional(Sequence(Tensor)) types up to IRv4.")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    24,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
        .TypeConstraint("V", OpSchema::all_tensor_types_ir12(), "All Tensor types up to IRv12.")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    23,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
        .TypeConstraint("V", OpSchema::all_tensor_types_ir11(), "All Tensor types up to IRv11.")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    21,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
        .TypeConstraint("V", OpSchema::all_tensor_types_ir10(), "All Tensor types up to IRv10.")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    19,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
        .TypeConstraint("V", OpSchema::all_tensor_types_ir9(), "All Tensor types up to IRv9.")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    16,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
        .TypeConstraint("V", OpSchema::all_tensor_types_ir4(), "All Tensor types up to IRv4.")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

// NOLINTNEXTLINE(misc-use-internal-linkage)
void ScanInferenceFunction_opset8(InferenceContext& ctx) {
  // NOTE:
  // The first input to Scan is sequence_lens. We skip that when processing
  // inputs in many places below, so the - 1 in multiple places is due to that.
  auto num_inputs = ctx.getNumInputs();
  auto num_scan_inputs = narrow<size_t>(getRequiredAttributeInt(ctx, "num_scan_inputs"));
  // Guard the subtraction below; opset 8's first input is sequence_lens, hence the + 1.
  if (num_scan_inputs + 1 > num_inputs) {
    fail_shape_inference(
        "num_scan_inputs (",
        num_scan_inputs,
        ") plus the sequence_lens input cannot exceed the number of Scan inputs (",
        num_inputs,
        ").");
  }
  auto num_loop_state_vars = num_inputs - 1 - num_scan_inputs;

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs);

  std::vector<const TypeProto*> subgraph_input_types;

  TensorShapeProto_Dimension batch_size_dim;
  TensorShapeProto_Dimension sequence_len_dim;

  for (size_t i = 1; i < num_inputs; ++i) {
    bool is_loop_state_var = (i - 1) < num_loop_state_vars;
    bool has_shape = hasInputShape(ctx, i);
    const auto* const input_type = ctx.getInputType(i);

    // Enforce type constraint for inputs
    if (!input_type || !input_type->has_tensor_type()) {
      fail_type_inference("Scan input ", i, " was not a tensor.");
    }

    if (is_loop_state_var) {
      // If it's a loop state variable we can propagate type and shape 1:1 to
      // the matching Scan output.
      // We can also pass through the type and shape to the subgraph but need to
      // remove the batch size dimension from the shape.
      propagateElemTypeFromInputToOutput(ctx, i, i - 1);

      if (has_shape) {
        propagateShapeFromInputToOutput(ctx, i, i - 1);

        // remove batch size dimension and add to subgraph_input_types
        temporary_type_protos.emplace_back(RemoveDimensionsFromShape(*input_type, 1));
        subgraph_input_types.emplace_back(&temporary_type_protos.back());
      } else {
        subgraph_input_types.emplace_back(input_type);
      }
    } else {
      // For other inputs there is no fixed relationships to the Scan outputs,
      // so we don't propagate type/shape information.
      // We can pass through the type and shape to the subgraph inputs but need
      // to remove the batch size and sequence length dimensions from the shape.
      if (has_shape) {
        // remove batch size and sequence length dimensions and add to
        // subgraph_input_types
        temporary_type_protos.emplace_back(RemoveDimensionsFromShape(*input_type, 2));
        subgraph_input_types.emplace_back(&temporary_type_protos.back());

        // update batch_size and sequence_len if a value is available
        const auto& shape = input_type->tensor_type().shape();
        if (shape.dim_size() > 2) {
          const auto& dims = shape.dim();
          mergeInDimensionInfo(dims.Get(0), batch_size_dim, 0);
          mergeInDimensionInfo(dims.Get(1), sequence_len_dim, 1);
        }
      } else {
        subgraph_input_types.emplace_back(input_type);
      }
    }
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.emplace_back(ctx.getInputData(i));
    }

    output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();
    if (output_types.size() != num_outputs) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          output_types.size(),
          " outputs. Expected ",
          num_outputs);
    }

    // propagate type/shape information for loop state variables and outputs
    for (size_t i = 0; i < num_outputs; ++i) {
      const bool is_loop_state_var = i < num_loop_state_vars;
      const auto* const subgraph_output_type = output_types[i];
      auto* scan_output_type = ctx.getOutputType(i);

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference("Scan 'body' subgraph outputs should all be tensors but output ", i, " was not");
      }

      // propagate output type. loop state vars were done in the above code.
      if (!is_loop_state_var) {
        scan_output_type->mutable_tensor_type()->set_elem_type(subgraph_output_type->tensor_type().elem_type());
      }

      // propagate shape
      if (subgraph_output_type->tensor_type().has_shape()) {
        // we need to add in the batch size and sequence length values if
        // available before merging with any existing info. Create a copy of the
        // inferred type info from the subgraph to do that.
        TypeProto inferred_type(*subgraph_output_type);
        auto* mutable_inferred_tensor_type = inferred_type.mutable_tensor_type();
        auto* mutable_inferred_shape = mutable_inferred_tensor_type->mutable_shape();

        mutable_inferred_shape->clear_dim();
        *mutable_inferred_shape->add_dim() = batch_size_dim;

        if (!is_loop_state_var) {
          *mutable_inferred_shape->add_dim() = sequence_len_dim;
        }

        for (const auto& dim : subgraph_output_type->tensor_type().shape().dim()) {
          (*mutable_inferred_shape->add_dim()) = dim;
        }

        auto* mutable_scan_output_tensor_type = scan_output_type->mutable_tensor_type();

        mergeInShapeInfo(*mutable_inferred_tensor_type, *mutable_scan_output_tensor_type);
      }
    }
  }
}

static int handle_negative_axis_validate_opset9(const std::string& attrib, int axis, int rank) {
  if (-rank > axis || axis >= rank) {
    fail_shape_inference(attrib, " axis value ", axis, " is invalid for a tensor of rank ", rank);
  }
  return (axis >= 0 ? axis : axis + rank);
}

static void ScanInferenceFunction_opset9(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  auto num_scan_inputs = narrow<size_t>(getRequiredAttributeInt(ctx, "num_scan_inputs"));
  auto num_outputs = ctx.getNumOutputs();
  auto num_loop_state_vars = ValidateScanCountsAndGetNumLoopStateVars(num_inputs, num_scan_inputs, num_outputs);
  auto num_scan_outputs = num_outputs - num_loop_state_vars;

  std::vector<int64_t> axes, output_axes;
  if (getRepeatedAttribute(ctx, "scan_input_axes", axes)) {
    if (axes.size() != num_scan_inputs) {
      fail_shape_inference(
          "Number of scan input axes specified (",
          axes.size(),
          ") is not equal to number of scan inputs (",
          num_scan_inputs,
          ").");
    }
  } else {
    axes.insert(axes.end(), num_scan_inputs, 0);
  }

  if (getRepeatedAttribute(ctx, "scan_output_axes", output_axes)) {
    if (output_axes.size() != num_scan_outputs) {
      fail_shape_inference(
          "Number of scan output axes specified (",
          output_axes.size(),
          ") is not equal to number of scan outputs (",
          num_scan_outputs,
          ").");
    }
  } else {
    output_axes.insert(output_axes.end(), num_scan_outputs, 0);
  }

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs);

  std::vector<const TypeProto*> subgraph_input_types;

  TensorShapeProto_Dimension sequence_len_dim;

  for (size_t i = 0; i < num_inputs; ++i) {
    bool is_loop_state_var = i < num_loop_state_vars;
    bool has_shape = hasInputShape(ctx, i);
    const auto* const input_type = ctx.getInputType(i);

    // Enforce type constraint for inputs
    if (!input_type || !input_type->has_tensor_type()) {
      fail_type_inference("Scan input ", i, " was not a tensor.");
    }

    if (is_loop_state_var) {
      // If it's a loop state variable we can propagate type and shape 1:1 to
      // the matching Scan output.
      // We can also pass through the type and shape to the subgraph but need to
      // remove the batch size dimension from the shape.
      propagateElemTypeFromInputToOutput(ctx, i, i);
      if (has_shape)
        propagateShapeFromInputToOutput(ctx, i, i);

      subgraph_input_types.emplace_back(input_type);
    } else {
      // For other inputs there is no fixed relationships to the Scan outputs,
      // so we don't propagate type/shape information.
      // We can pass through the type and shape to the subgraph inputs but
      // need to remove the sequence length dimensions from the shape.
      if (has_shape) {
        const auto& shape = input_type->tensor_type().shape();

        // remove sequence length dimensions and add to subgraph_input_types
        int axis = static_cast<int>(axes[i - num_loop_state_vars]);
        axis = handle_negative_axis_validate_opset9("scan_input_axes", axis, shape.dim_size());

        // update sequence_len if a value is available

        const auto& dims = shape.dim();
        mergeInDimensionInfo(dims.Get(axis), sequence_len_dim, 1);

        temporary_type_protos.emplace_back(RemoveIthDimensionFromShape(*input_type, axis));
        subgraph_input_types.emplace_back(&temporary_type_protos.back());

      } else {
        subgraph_input_types.emplace_back(input_type);
      }
    }
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      // ctx.getInputData(i), the input to scan, does not represent the input to
      // scan body. So, we pass in null, to represent an unknown value.
      input_data.emplace_back(nullptr);
    }

    output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!output_types.empty()) {
    if (output_types.size() != num_outputs) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          output_types.size(),
          " outputs. Expected ",
          num_outputs);
    }

    // propagate type/shape information for loop state variables and outputs
    for (size_t i = 0; i < num_outputs; ++i) {
      const bool is_loop_state_var = i < num_loop_state_vars;
      const auto* const subgraph_output_type = output_types[i];
      auto* scan_output_type = ctx.getOutputType(i);
      auto* mutable_scan_output_tensor_type = scan_output_type->mutable_tensor_type();

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference("Scan 'body' subgraph outputs should all be tensors but output ", i, " was not");
      }
      const auto& subgraph_output_tensor_type = subgraph_output_type->tensor_type();

      if (is_loop_state_var) {
        // merge shape; type already propagated
        mergeInShapeInfo(subgraph_output_tensor_type, *mutable_scan_output_tensor_type);
      } else {
        scan_output_type->mutable_tensor_type()->set_elem_type(subgraph_output_tensor_type.elem_type());

        // propagate shape
        if (subgraph_output_tensor_type.has_shape()) {
          // infer shape of scan-output from the shape of scan-output-element
          // by adding sequence-length at the correct axis position
          const TensorShapeProto& subgraph_output_shape = subgraph_output_tensor_type.shape();
          TensorShapeProto inferred_shape;

          auto subgraph_output_rank = subgraph_output_shape.dim_size();
          auto output_rank = subgraph_output_rank + 1;
          int output_axis = static_cast<int>(output_axes[i - num_loop_state_vars]);
          output_axis = handle_negative_axis_validate_opset9("scan_output_axes", output_axis, output_rank);

          for (int j = 0; j < output_axis; ++j)
            *(inferred_shape.add_dim()) = subgraph_output_shape.dim(j);
          *(inferred_shape.add_dim()) = sequence_len_dim;
          for (int j = output_axis; j < subgraph_output_rank; ++j)
            *(inferred_shape.add_dim()) = subgraph_output_shape.dim(j);

          // Merge inferred shape with existing shape information
          mergeInShapeInfo(inferred_shape, *mutable_scan_output_tensor_type);
        }
      }
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    8,
    OpSchema()
        .SetDoc(kDoc_scan_ver8)
        .Input(
            0,
            "sequence_lens",
            "Optional tensor specifying lengths of the sequences in a batch. "
            "If this input is not specified, all sequences are assumed to be of "
            "the maximum sequence length (the dimension of the sequence axis of "
            "the scan_input tensors).",
            "I",
            OpSchema::Optional)
        .Input(
            1,
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
            "directions",
            "An optional list of M flags. The i-th element of the list specifies the direction "
            "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
            "indicates reverse direction. "
            "If omitted, all scan_input tensors will be scanned in the forward direction.",
            AttributeProto::INTS,
            false)
        .TypeConstraint("I", {types::Int64}, "Int64 tensor")
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction_opset8));

static void LoopInferenceFunction_opset8(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  auto num_loop_state_vars = num_inputs - 2; // skip 'M' and 'cond'

  std::vector<const TypeProto*> subgraph_input_types;

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs - 2);

  // create TypeProto to validate iteration number type is the same as the
  // optional 'M' input for max iterations.
  TypeProto iter_num_type;
  iter_num_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  subgraph_input_types.emplace_back(&iter_num_type);

  // 'cond'
  subgraph_input_types.emplace_back(ctx.getInputType(1));

  // loop state value types get propagated to outputs, but shape may change
  // across iterations so don't propagate it to the outputs and don't pass it
  // into the subgraph inferencing
  for (size_t i = 2; i < num_inputs; ++i) {
    propagateElemTypeFromInputToOutput(ctx, i, i - 2);

    // copy so we can remove the shape before passing to the subgraph
    // inferencing
    temporary_type_protos.emplace_back(*ctx.getInputType(i));
    auto& input_type = temporary_type_protos.back();
    input_type.mutable_tensor_type()->clear_shape();

    subgraph_input_types.emplace_back(&input_type);
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> subgraph_output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.emplace_back(nullptr); // iteration number
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.emplace_back(ctx.getInputData(i));
    }

    subgraph_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();

    // subgraph outputs the condition value first but that is only used
    // internally and not returned by Loop.
    if (subgraph_output_types.size() != num_outputs + 1) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs + 1);
    }

    // check loop state values match. we should already have type/shape info
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto* const subgraph_output_type = subgraph_output_types[i + 1]; // skip 'cond'
      auto* loop_output_type = ctx.getOutputType(i);

      const bool is_loop_state_var = i < num_loop_state_vars;

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Loop 'body' subgraph outputs should all be tensors but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      // if there's an existing type check it matches. otherwise propagate
      propagateElemTypeWithValidation(subgraph_output_type, loop_output_type);

      if (is_loop_state_var) {
        // shape may change across iterations so ignore.
      } else {
        // propagate shape
        if (subgraph_output_type->tensor_type().has_shape()) {
          // per iteration output. first dimension will be number of iterations
          // but we don't know that value yet
          TypeProto inferred_type(*subgraph_output_type);
          auto* mutable_inferred_tensor_type = inferred_type.mutable_tensor_type();
          auto* mutable_inferred_shape = mutable_inferred_tensor_type->mutable_shape();

          mutable_inferred_shape->clear_dim();

          // add empty dimension for number of iterations
          mutable_inferred_shape->add_dim();

          // add dimensions from subgraph output shape
          for (const auto& dim : subgraph_output_type->tensor_type().shape().dim()) {
            (*mutable_inferred_shape->add_dim()) = dim;
          }

          mergeInShapeInfo(*mutable_inferred_tensor_type, *loop_output_type->mutable_tensor_type());
        }
      }
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    1,
    OpSchema()
        .SetDoc(kDoc_Loop_ver1)
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
            false)
        .Output(
            0,
            "v_final_and_scan_outputs",
            "Final N loop carried dependency values then K scan_outputs",
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
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction_opset8));

static void LoopInferenceFunction_opset11(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  auto num_loop_state_vars = num_inputs - 2; // skip 'M' and 'cond'

  std::vector<const TypeProto*> subgraph_input_types;

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs - 2);

  // create TypeProto to validate iteration number type is the same as the
  // optional 'M' input for max iterations.
  TypeProto iter_num_type;
  iter_num_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  subgraph_input_types.emplace_back(&iter_num_type);

  // 'cond'
  subgraph_input_types.emplace_back(ctx.getInputType(1));

  // loop state value types get propagated to outputs, but shape may change
  // across iterations so don't propagate it to the outputs and don't pass it
  // into the subgraph inferencing
  for (size_t i = 2; i < num_inputs; ++i) {
    propagateElemTypeFromInputToOutput(ctx, i, i - 2);

    // copy so we can remove the shape before passing to the subgraph
    // inferencing
    temporary_type_protos.emplace_back(*ctx.getInputType(i));
    auto& input_type = temporary_type_protos.back();
    input_type.mutable_tensor_type()->clear_shape();

    subgraph_input_types.emplace_back(&input_type);
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> subgraph_output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.emplace_back(nullptr); // iteration number
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.emplace_back(ctx.getInputData(i));
    }

    subgraph_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();

    // subgraph outputs the condition value first but that is only used
    // internally and not returned by Loop.
    if (subgraph_output_types.size() != num_outputs + 1) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs + 1);
    }

    // check loop state values match. we should already have type/shape info
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto* const subgraph_output_type = subgraph_output_types[i + 1]; // skip 'cond'
      auto* loop_output_type = ctx.getOutputType(i);

      const bool is_loop_state_var = i < num_loop_state_vars;

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Loop 'body' subgraph outputs should all be tensors but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      // if there's an existing type check it matches. otherwise propagate
      propagateElemTypeWithValidation(subgraph_output_type, loop_output_type);

      if (is_loop_state_var) {
        // shape may change across iterations so ignore.
      } else {
        // propagate shape
        if (subgraph_output_type->tensor_type().has_shape()) {
          // per iteration output. first dimension will be number of iterations
          // but we don't know that value yet
          TypeProto inferred_type(*subgraph_output_type);
          auto* mutable_inferred_tensor_type = inferred_type.mutable_tensor_type();
          auto* mutable_inferred_shape = mutable_inferred_tensor_type->mutable_shape();

          mutable_inferred_shape->clear_dim();

          // add empty dimension for number of iterations
          mutable_inferred_shape->add_dim();

          // add dimensions from subgraph output shape
          for (const auto& dim : subgraph_output_type->tensor_type().shape().dim()) {
            (*mutable_inferred_shape->add_dim()) = dim;
          }

          mergeInShapeInfo(*mutable_inferred_tensor_type, *loop_output_type->mutable_tensor_type());
        }
      }
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    11,
    OpSchema()
        .SetDoc(kDoc_Loop_ver11)
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
            "Final N loop carried dependency values then K scan_outputs",
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
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    9,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
            "be used as the scan axis for every scan_input.",
            AttributeProto::INTS,
            false)
        .Attr(
            "scan_output_axes",
            "An optional list of K flags. The i-th element of the list specifies the axis "
            "for the i-th scan_output. The scan outputs are accumulated along the specified "
            "axis. If omitted, 0 will be used as the scan axis for every scan_output.",
            AttributeProto::INTS,
            false)
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction_opset9));

static void IfInferenceFunction_opset1(InferenceContext& ctx) {
  // there are no inputs so we just need to run the subgraph inferencing for
  // then/else subgraphs and apply those to the outputs.
  std::vector<const TypeProto*> subgraph_input_types; // none
  std::vector<const TensorProto*> input_data; // none

  std::vector<const TypeProto*> then_output_types;
  std::vector<const TypeProto*> else_output_types;

  // Run inferencing on the subgraph
  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("then_branch");
  if (graphInferencer) {
    then_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  graphInferencer = ctx.getGraphAttributeInferencer("else_branch");
  if (graphInferencer) {
    else_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  auto num_outputs = ctx.getNumOutputs();
  auto num_then_outputs = then_output_types.size();
  auto num_else_outputs = else_output_types.size();

  // the output types for then and else should be the same
  if (num_then_outputs != num_else_outputs) {
    fail_type_inference(
        "then_branch and else_branch produce different number of outputs. ",
        num_then_outputs,
        " != ",
        num_else_outputs);
  }

  if (num_then_outputs != num_outputs) {
    fail_type_inference("If node has ", num_outputs, " but subgraphs produce ", num_then_outputs);
  }

  for (size_t i = 0, end = then_output_types.size(); i < end; ++i) {
    const auto* const then_output = then_output_types[i];
    const auto* const else_output = else_output_types[i];

    if (then_output->value_case() != else_output->value_case()) {
      fail_type_inference(
          "Mismatched type for output ", i, " then=", then_output->value_case(), " else=", else_output->value_case());
    }

    auto* if_output = ctx.getOutputType(i);
    *if_output = *then_output;

    if (then_output->has_tensor_type()) {
      auto then_elem_type = then_output->tensor_type().elem_type();
      auto else_elem_type = else_output->tensor_type().elem_type();

      if (then_elem_type != else_elem_type) {
        fail_type_inference(
            "Mismatched tensor element type for output ", i, " then=", then_elem_type, " else=", else_elem_type);
      }

      // merge the 'else' shape information to check it's consistent and
      // augment the 'if' output if possible
      mergeInShapeInfo(else_output->tensor_type(), *if_output->mutable_tensor_type());
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    If,
    1,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
        .Input(0, "cond", "Condition for the if. The tensor must contain a single element.", "B")
        .Output(
            0,
            "outputs",
            "Values that are live-out to the enclosing scope. The return values in "
            "the `then_branch` and `else_branch` must be of the same shape and same "
            "data type.",
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
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction_opset1));

static void IfInferenceFunction_opset11(InferenceContext& ctx) {
  // there are no inputs so we just need to run the subgraph inferencing for
  // then/else subgraphs and apply those to the outputs.
  std::vector<const TypeProto*> subgraph_input_types; // none
  std::vector<const TensorProto*> input_data; // none

  std::vector<const TypeProto*> then_output_types;
  std::vector<const TypeProto*> else_output_types;

  // Run inferencing on the subgraph
  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("then_branch");
  if (graphInferencer) {
    then_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  graphInferencer = ctx.getGraphAttributeInferencer("else_branch");
  if (graphInferencer) {
    else_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  auto num_outputs = ctx.getNumOutputs();
  auto num_then_outputs = then_output_types.size();
  auto num_else_outputs = else_output_types.size();

  // the output types for then and else should be the same
  if (num_then_outputs != num_else_outputs) {
    fail_type_inference(
        "then_branch and else_branch produce different number of outputs. ",
        num_then_outputs,
        " != ",
        num_else_outputs);
  }

  if (num_then_outputs != num_outputs) {
    fail_type_inference("If node has ", num_outputs, " but subgraphs produce ", num_then_outputs);
  }

  for (size_t i = 0, end = then_output_types.size(); i < end; ++i) {
    const auto* const then_output = then_output_types[i];
    const auto* const else_output = else_output_types[i];

    if (then_output->value_case() != else_output->value_case()) {
      fail_type_inference(
          "Mismatched type for output ", i, " then=", then_output->value_case(), " else=", else_output->value_case());
    }

    auto* if_output = ctx.getOutputType(i);
    *if_output = *then_output;

    if (then_output->has_tensor_type()) {
      auto then_elem_type = then_output->tensor_type().elem_type();
      auto else_elem_type = else_output->tensor_type().elem_type();

      if (then_elem_type != else_elem_type) {
        fail_type_inference(
            "Mismatched tensor element type for output ", i, " then=", then_elem_type, " else=", else_elem_type);
      }

      UnionShapeInfo(else_output->tensor_type().shape(), *if_output->mutable_tensor_type());
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    If,
    11,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction_opset11));

static void IfInferenceFunction_opset13(InferenceContext& ctx) {
  // there are no inputs so we just need to run the subgraph inferencing for
  // then/else subgraphs and apply those to the outputs.
  std::vector<const TypeProto*> subgraph_input_types; // none
  std::vector<const TensorProto*> input_data; // none

  std::vector<const TypeProto*> then_output_types;
  std::vector<const TypeProto*> else_output_types;

  // Run inferencing on the subgraph
  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("then_branch");
  if (graphInferencer) {
    then_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  graphInferencer = ctx.getGraphAttributeInferencer("else_branch");
  if (graphInferencer) {
    else_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  auto num_outputs = ctx.getNumOutputs();
  auto num_then_outputs = then_output_types.size();
  auto num_else_outputs = else_output_types.size();

  // the output types for then and else should be the same
  if (num_then_outputs != num_else_outputs) {
    fail_type_inference(
        "then_branch and else_branch produce different number of outputs. ",
        num_then_outputs,
        " != ",
        num_else_outputs);
  }

  if (num_then_outputs != num_outputs) {
    fail_type_inference("If node has ", num_outputs, " but subgraphs produce ", num_then_outputs);
  }

  for (size_t i = 0, end = then_output_types.size(); i < end; ++i) {
    const auto* const then_output = then_output_types[i];
    const auto* const else_output = else_output_types[i];

    auto* if_output = ctx.getOutputType(i);
    *if_output = *then_output;

    UnionTypeInfo(*else_output, *if_output);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    If,
    13,
    OpSchema()
        .SetDoc(kDoc_If_ver25)
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
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "All Tensor and Sequence types")
        .TypeConstraint("B", {types::Bool}, "Only bool")
        .TypeAndShapeInferenceFunction(IfInferenceFunction_opset13));

static void LoopInferenceFunction_opset13(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  assert(num_inputs >= 2);
  auto num_loop_state_vars = num_inputs - 2; // skip 'M' and 'cond'

  std::vector<const TypeProto*> subgraph_input_types;
  subgraph_input_types.reserve(num_inputs);

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs - 2);

  // create TypeProto to validate iteration number type is the same as the
  // optional 'M' input for max iterations.
  TypeProto iter_num_type;
  iter_num_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  subgraph_input_types.emplace_back(&iter_num_type);

  // 'cond'
  subgraph_input_types.emplace_back(ctx.getInputType(1));

  // loop state value types get propagated to outputs, but shape may change
  // across iterations so don't propagate it to the outputs and don't pass it
  // into the subgraph inferencing
  for (size_t i = 2; i < num_inputs; ++i) {
    propagateElemTypeFromInputToOutput(ctx, i, i - 2);

    // copy so we can remove the shape before passing to the subgraph
    // inferencing
    temporary_type_protos.emplace_back(*ctx.getInputType(i));
    auto& input_type = temporary_type_protos.back();

    if (input_type.has_tensor_type()) {
      input_type.mutable_tensor_type()->clear_shape();
    } else if (input_type.has_sequence_type()) {
      auto& seq_type = *input_type.mutable_sequence_type();
      if (seq_type.has_elem_type() && seq_type.elem_type().has_tensor_type()) {
        seq_type.mutable_elem_type()->mutable_tensor_type()->clear_shape();
      }
    }

    subgraph_input_types.emplace_back(&input_type);
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> subgraph_output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.emplace_back(nullptr); // iteration number
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.emplace_back(ctx.getInputData(i));
    }

    subgraph_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();

    // subgraph outputs the condition value first but that is only used
    // internally and not returned by Loop.
    if (subgraph_output_types.size() != num_outputs + 1) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs + 1);
    }

    // check loop state values match. we should already have type/shape info
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto* const subgraph_output_type = subgraph_output_types[i + 1]; // skip 'cond'
      auto* loop_output_type = ctx.getOutputType(i);

      const bool is_loop_state_var = i < num_loop_state_vars;

      if (!subgraph_output_type->has_tensor_type() && !subgraph_output_type->has_sequence_type()) {
        fail_type_inference(
            "Loop 'body' subgraph outputs should all be tensors or sequences but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      if (!is_loop_state_var && !subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Loop 'body' subgraph scan outputs should all be tensors but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      // if there's an existing type check it matches. otherwise propagate
      propagateElemTypeWithValidation(subgraph_output_type, loop_output_type);

      if (is_loop_state_var) {
        // shape may change across iterations so ignore.
      } else {
        // propagate shape
        if (subgraph_output_type->tensor_type().has_shape()) {
          // per iteration output. first dimension will be number of iterations
          // but we don't know that value yet
          TypeProto inferred_type(*subgraph_output_type);
          auto* mutable_inferred_tensor_type = inferred_type.mutable_tensor_type();
          auto* mutable_inferred_shape = mutable_inferred_tensor_type->mutable_shape();

          mutable_inferred_shape->clear_dim();

          // add empty dimension for number of iterations
          mutable_inferred_shape->add_dim();

          // add dimensions from subgraph output shape
          for (const auto& dim : subgraph_output_type->tensor_type().shape().dim()) {
            (*mutable_inferred_shape->add_dim()) = dim;
          }

          mergeInShapeInfo(*mutable_inferred_tensor_type, *loop_output_type->mutable_tensor_type());
        }
      }
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    13,
    OpSchema()
        .SetDoc(kDoc_Loop_ver13)
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
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "All Tensor and Sequence types")
        .TypeConstraint("I", {types::Int64}, "tensor of int64, which should be a scalar.")
        .TypeConstraint("B", {types::Bool}, "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    11,
    OpSchema()
        .SetDoc(kDoc_scan_24)
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
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeAndShapeInferenceFunction(ScanInferenceFunction));

} // namespace ONNX_NAMESPACE
