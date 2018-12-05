// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;

void ScanInferenceFunctionOpset8(InferenceContext& ctx) {
  // NOTE:
  // The first input to Scan is sequence_lens. We skip that when processing
  // inputs in many places below, so the - 1 in multiple places is due to that.
  auto num_inputs = ctx.getNumInputs();
  auto num_scan_inputs =
      narrow_cast<size_t>(ctx.getAttribute("num_scan_inputs")->i());
  auto num_loop_state_vars = num_inputs - 1 - num_scan_inputs;

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs);

  std::vector<const TypeProto*> subgraph_input_types;

  TensorShapeProto_Dimension batch_size_dim;
  TensorShapeProto_Dimension sequence_len_dim;

  for (size_t i = 1; i < num_inputs; ++i) {
    bool is_loop_state_var = (i - 1) < num_loop_state_vars;
    bool has_shape = hasInputShape(ctx, i);
    const auto* input_type = ctx.getInputType(i);

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
        temporary_type_protos.push_back(
            RemoveDimensionsFromShape(*input_type, 1));
        subgraph_input_types.push_back(&temporary_type_protos.back());
      } else {
        subgraph_input_types.push_back(input_type);
      }
    } else {
      // For other inputs there is no fixed relationships to the Scan outputs,
      // so we don't propagate type/shape information.
      // We can pass through the type and shape to the subgraph inputs but need
      // to remove the batch size and sequence length dimensions from the shape.
      if (has_shape) {
        // remove batch size and sequence length dimensions and add to
        // subgraph_input_types
        temporary_type_protos.push_back(
            RemoveDimensionsFromShape(*input_type, 2));
        subgraph_input_types.push_back(&temporary_type_protos.back());

        // update batch_size and sequence_len if a value is available
        const auto& shape = input_type->tensor_type().shape();
        if (shape.dim_size() > 2) {
          const auto& dims = shape.dim();
          mergeInDimensionInfo(dims.Get(0), batch_size_dim, 0);
          mergeInDimensionInfo(dims.Get(1), sequence_len_dim, 1);
        }
      } else {
        subgraph_input_types.push_back(input_type);
      }
    }
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.push_back(ctx.getInputData(i));
    }

    output_types =
        graphInferencer->doInferencing(subgraph_input_types, input_data);
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
      auto* subgraph_output_type = output_types[i];
      auto* scan_output_type = ctx.getOutputType(i);

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Scan 'body' subgraph outputs should all be tensors but output ",
            i,
            " was not");
      }

      // propagate output type. loop state vars were done in the above code.
      if (!is_loop_state_var) {
        scan_output_type->mutable_tensor_type()->set_elem_type(
            subgraph_output_type->tensor_type().elem_type());
      }

      // propagate shape
      if (subgraph_output_type->tensor_type().has_shape()) {
        // we need to add in the batch size and sequence length values if
        // available before merging with any existing info. Create a copy of the
        // inferred type info from the subgraph to do that.
        TypeProto inferred_type(*subgraph_output_type);
        auto* mutable_inferred_tensor_type =
            inferred_type.mutable_tensor_type();
        auto* mutable_inferred_shape =
            mutable_inferred_tensor_type->mutable_shape();

        mutable_inferred_shape->clear_dim();
        *mutable_inferred_shape->add_dim() = batch_size_dim;

        if (!is_loop_state_var) {
          *mutable_inferred_shape->add_dim() = sequence_len_dim;
        }

        for (const auto& dim :
             subgraph_output_type->tensor_type().shape().dim()) {
          (*mutable_inferred_shape->add_dim()) = dim;
        }

        auto* mutable_scan_output_tensor_type =
            scan_output_type->mutable_tensor_type();

        mergeInShapeInfo(
            *mutable_inferred_tensor_type, *mutable_scan_output_tensor_type);
      }
    }
  }
}

static const char* scan_opset8_doc = R"DOC(
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops). All these tensors are required to
have the same shape in each iteration of the loop (a restriction imposed to enable efficient
memory allocation). Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs).

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The operation supports batching, and the batch-axis is required to be 0.
When multiple scan_input tensors are used, they must all have the same batch-size,
and they must all have the same maximum-sequence-length (the dimensionality of the
sequence axis or scan axis). The sequence axis or scan axis is required to be 1.

The operation has an optional sequence_lens input (of shape [BATCH_SIZE]) to
allow variable length sequences of length <= the maximum-sequence-length. If this
input is not specified, all sequences are assumed to be of length equal to
maximum-sequence-length. For variable length input sequences, the scan_outputs
will consist of a sequence of same length as the input, padded to the
maximum-sequence-length.

The optional attribute directions can be used to scan a sequence in the reverse direction.
If this attribute is omitted, all sequences are scanned in the forward direction.
A bidirectional scan be performed by specifying the same tensor input twice in the
scan_inputs, once with a forward direction, and once with a backward direction.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body
    > (sequence_lengths, init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // T.shape[0] denotes the batch-size of T
    // The batch-size of scan_1, ..., scan_m are all required to be equal
    batch_size = scan_1.shape[0];

    // scan_i.shape[1] denotes the (max) sequence-length of scan_i
    // scan_i.shape[1] is required to be equal to scan_j.shape[1] for all i,j.
    max_sequence_length = scan_1.shape[1];

    for (int batch = 0; batch < batch_size; ++batch) {
        // initialize state-variables
        st_1 = init_1; ... st_n = init_n;
        // initialize scan-output variables: [] denotes an empty tensor
        scan_out_1 = []; ...; scan_out_k = [];
        // identify number of iterations:
        N = (sequence_lengths specified) ? sequence_lengths[batch] : max_sequence_length;

        // execute loop
        for (int t = 0; t < N; ++t) {
            // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
            // of rank one less than T obtained by indexing T at position t along axis k.
            si_1 = (scan_1<axis=0>[batch])<axis=1>[t];
            ... ;
            si_m = (scan_m<axis=0>[batch])<axis=1>[t];
            // execute loop-body
            st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
            // accumulate the scan-output elements
            scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
        }
        // accumulate the outputs for this batch:
        bst_1[batch] = st_1; ..., bst_n[batch] = st_n;
        // Note scan-outputs will have size max_sequence_length, but only first N values will be meaningful.
        // The remaining values have an undefined value.
        b_scan_out_1[batch] = scan_out_1; ...; b_scan_out_k[batch] = scan_out_k;
    }
    return bst_1, ..., bst_n, b_scan_out_1, ..., b_scan_out_k;



*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ... 
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1]("", %H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
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
        .SetDoc(scan_opset8_doc)
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
        .Attr(
            "num_scan_inputs",
            "An attribute specifying the number of scan_inputs M. ",
            AttributeProto::INT,
            true)
        .Attr(
            "directions",
            "An optional list of M flags. The i-th element of the list specifies the direction "
            "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
            "indicates reverse direction. "
            "If omitted, all scan_input tensors will be scanned in the forward direction.",
            AttributeProto::INTS,
            false)
        .TypeConstraint("I", {"tensor(int64)"}, "Int64 tensor")
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeAndShapeInferenceFunction(ScanInferenceFunctionOpset8));

} // namespace ONNX_NAMESPACE
