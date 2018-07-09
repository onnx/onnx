// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;

ONNX_OPERATOR_SET_SCHEMA(
    If,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .SetDoc("If conditional")
        .Input(0, "cond", "Condition for the if", "B")
        .Output(
            0,
            "outputs",
            "Values that are live-out to the enclosing scope.",
            "V",
            OpSchema::Variadic)
        .Attr(
            "then_branch",
            "Graph to run if condition is true. Has N outputs: values you wish to "
            "be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the else_branch.",
            AttributeProto::GRAPH,
            true)
        .Attr(
            "else_branch",
            "Graph to run if condition is false. Has N outputs: values you wish to"
            " be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the then_branch.",
            AttributeProto::GRAPH,
            true)
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("B", {"tensor(bool)"}, "Only bool"));

static const char* scan_loop_ver1_doc = R"DOC(
ScanLoop can be used to iterate over (specified axes of) one or more scan_input tensors,
constructing zero or more scan_output tensors. Other tensors can be used to carry a state
when iterating from one element to another (referred to as loop-carried dependences
below). All these tensors are required to have the same shape in each iteration of the loop.

The behavior of
   ScanLoop <scan_axes = [axis_1, ..., axis_m], body = loop-body> (init_1, ..., init_n, scan_1, ..., scan_m)
is equivalent to the following pseudo-code:

	{
		// initialize state-variables
		st_1 = init_1; ... st_n = init_n;
		// initialize scan-output variables: [] denotes an empty tensor
		scan_out_1 = []; ...; scan_out_k = [];
		// identify number of iterations: T.shape[i] denotes the size of T's i-th axis/dimension
		dim_1 = scan_1.shape[axis_1]; ... ; dim_m = scan_m.shape[axis_m];
		N = min(dim_1, ..., dim_m);
		// execute loop
		for (int t = 0; t < N; ++t) {
			// generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
			// of rank one less than T obtained by indexing T at position t along axis k.
			si_1 = scan_1<axis=axis_1>[t]; ... ; si_m = scan_m<axis=axis_m>[t];
			// execute loop-body
			st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
			// accumulate the scan-output elements
			scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
		}
		return st_1, ..., st_n, scan_out_1, ..., scan_out_k
	}

*Sample usage: Encoding RNN using a ScanLoop*
The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly refers
to the names %Wi, %Ri, %Wbi, abd %Rbi defined in the outer graph.

    graph rnn-encoding {
      %H_0 = ... 
      %X = ...
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %Y_h, %Y = ScanLoop[body = <graph rnn-cell-1>, scan_axes=[0]](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
	Scan,
	8,
	OpSchema()
	.SetSupportLevel(SupportType::EXPERIMENTAL)
	.SetDoc(scan_loop_ver1_doc)
	.Input(
		0,
		"initial_state_and_scan_inputs",
		"Initial values of the loop's N state variables followed by M scan_inputs",
		"V",
		OpSchema::Variadic)
	.Output(
		0,
		"final_state_and_scan_outputs",
		"Final values of the loop's N state variables followed by K scan_outputs",
		"V",
		OpSchema::Variadic)
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
	.Attr(
		"scan_axes",
		"A list of M axes. The i-th element of the list specifies the axis "
		"to be scanned for the i-th scan_input tensor.",
		AttributeProto::INTS
	)
	.TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types"));

static const char* scan_while_loop_ver1_doc = R"DOC(
ScanWhileLoop extends ScanLoop with a while-condition that allows the loop to
terminate earlier (before the scan-input tensors are completely scanned).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ScanWhile,
    8,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .SetDoc(scan_while_loop_ver1_doc)
	    .Input(
		    0,
		   "cond",
		   "Initial value of loop continuation condition. If this value is true "
		   "the loop will execute at least one time. If this value is false "
		   "the loop will execute zero times.",
		   "bool")
		.Input(
			1,
			"initial_state_and_scan_inputs",
			"Initial values of the loop's N state variables followed by M scan_inputs",
			"V",
			OpSchema::Variadic)
		.Output(
			0,
			"final_state_and_scan_outputs",
			"Final values of the loop's N state variables followed by K scan_outputs",
			"V",
			OpSchema::Variadic)
        .Attr(
            "body",
            "The graph run each iteration. It has N+M inputs: "
            "(loop state variables..., scan_input_elts...). It has 1+N+K outputs: "
            "(condition, loop cstate variables..., scan_output_elts...). Each "
            "scan_output is created by concatenating the value of the specified "
            "scan_output_elt at the end of each iteration of the loop. It is an error"
            " if the dimensions of these values change across loop iterations.",
            AttributeProto::GRAPH,
            true)
        .Attr(
            "scan_axes",
            "A list of M axes. The i-th element of the list specifies the axis "
            "to be scanned for the i-th scan_input tensor.",
            AttributeProto::INTS
        )
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types"));


} // namespace ONNX_NAMESPACE