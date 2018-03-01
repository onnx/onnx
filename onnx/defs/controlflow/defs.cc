// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace ONNX_NAMESPACE;

using SupportType = ONNX_NAMESPACE::OpSchema::SupportType;

ONNX_OPERATOR_SCHEMA(If)
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
    .TypeConstraint("B", {"tensor(bool)"}, "Only bool");

ONNX_OPERATOR_SCHEMA(Loop)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]
      %keepgoing[BOOL, scalar]
      %b[INT32, scalar]
    ) {
      %my_local = Add(%a, %b)
      %b_out = Sub(%a, %b)
      %keepgoing_out = Greater(%my_local, %b_out)
      %user_defined_vals = Add(%b, %b)
      return %keepgoing_out, %b_out, %user_defined_vals
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      for (int i=0; i < max_trip_count && keepgoing; ++i) {
        /* User-defined code (loop body) */
        int my_local = a + b; // Reading values in the enclosing scope is fine
        b = a - b; // writes fine if we specify b as a loop-carried dependency
        keepgoing = my_local > b; // keepgoing is a loop-carried dependency
        user_defined_vals[i] = b + b;
        /* End user-defined code */
      }
      // my_local = 123; // Can't do this. my_local was defined in the the body

      // These below values are live-out from the loop and therefore accessible
      b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable a here) are in scope and can
   be referenced in the inputs of the loop.
2) Any variables which you wish to make available in the enclosing scope (i.e.
   the variables b and keepgoing) must be declared as either loop-carried
   dependencies (both at the op inputs and output and at the body net input and
   output) or scan_outputs.
3) Values created in the body cannot be accessed in the enclosing scope.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).
Concretely, the (possibly transformed) scan_outputs are referenced by the
subsequent layer as a LoopIndexTensor operating on a value in scope, not
necessarily a loop-carried dependency. Backends can recognize this pattern and
are permitted to schedule the execution of the multi-layer network in a
pipelined/"wavefront" fashion.

)DOC")
    .Input(
        0,
        "M",
        "A maximum trip-count for the loop specified at runtime. Optional."
        " pass empty string to skip.",
        "I")
    .Input(
        1,
        "cond",
        "A boolean termination condition. Pass empty string to skip.",
        "B")
    .Input(
        2,
        "v_initial",
        "The initial values of any loop-carried dependencies (values that "
        "change across loop iterations)",
        "V",
        OpSchema::Variadic)
    .Output(
        0,
        "v_final_and_scan_outputs",
        "Final N loop carried dependency values then K scan_outputs",
        "V",
        OpSchema::Variadic)
    .Attr(
        "body",
        "The graph run each iteration. It has 2+N inputs: (iteration_num, "
        "condition, loop carried dependencies...). It has 1+N+K outputs: "
        "(condition, loop carried dependencies..., scan_outputs...). Each "
        "scan_output is created by concatenating the value of the specified "
        "output value at the end of each iteration of the loop. It is an error"
        " if the dimensions of these values change across loop iterations.",
        AttributeProto::GRAPH,
        true)
    .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
    .TypeConstraint("I", {"int64"}, "Only int64")
    .TypeConstraint("B", {"bool"}, "Only bool");

ONNX_OPERATOR_SCHEMA(LoopIndexTensor)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(
        "This is a special operator only valid inside the loop that supports "
        "the common case behavior of accessing the correct element of the input"
        " sequence in an RNN. This operator MUST be directly given the passed-"
        "in iteration number to the body of a Loop graph. This signals to back-"
        "ends that this is a direct indexing operation, with no transforms "
        "applied to the index.")
    .Input(0, "T", "Tensor to be indexed (has N dimensions)", "T")
    .Input(1, "loop_idx", "Loop index provided as input to the body graph", "I")
    .Attr(
        "axis",
        "Axis on which to index",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Output(0, "O", "Tensor of N - 1 dims that is a sub tensor of T", "T")
    .TypeConstraint("T", OpSchema::all_tensor_types(), "All Tensor types")
    .TypeConstraint("I", {"int32"}, "Indices");
