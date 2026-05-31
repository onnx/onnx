// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/controlflow/utils.h"

#include <string>
#include <vector>

#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

void ClearShape(TypeProto& input_type) {
  if (input_type.has_tensor_type()) {
    input_type.mutable_tensor_type()->clear_shape();
  } else if (input_type.has_sequence_type()) {
    auto& seq_type = *input_type.mutable_sequence_type();
    if (seq_type.has_elem_type()) {
      ClearShape(*(seq_type.mutable_elem_type()));
    }
  } else if (input_type.has_optional_type()) {
    auto& opt_type = *input_type.mutable_optional_type();
    if (opt_type.has_elem_type()) {
      ClearShape(*(opt_type.mutable_elem_type()));
    }
  }
}

void IfInferenceFunction(InferenceContext& ctx) {
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

void LoopInferenceFunction(InferenceContext& ctx) {
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
  subgraph_input_types.push_back(&iter_num_type);

  // 'cond'
  subgraph_input_types.push_back(ctx.getInputType(1));

  // loop state value types get propagated to outputs, but shape may change
  // across iterations so don't propagate it to the outputs and don't pass it
  // into the subgraph inferencing
  for (size_t i = 2; i < num_inputs; ++i) {
    propagateElemTypeFromInputToOutput(ctx, i, i - 2);

    // copy so we can remove the shape before passing to the subgraph
    // inferencing
    temporary_type_protos.push_back(*ctx.getInputType(i));
    auto& input_type = temporary_type_protos.back();

    ClearShape(input_type);
    subgraph_input_types.push_back(&input_type);
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> subgraph_output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.push_back(nullptr); // iteration number
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.push_back(ctx.getInputData(i));
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

      if (!subgraph_output_type->has_tensor_type() && !subgraph_output_type->has_sequence_type() &&
          !subgraph_output_type->has_optional_type()) {
        fail_type_inference(
            "Loop 'body' subgraph outputs should all be tensors or sequences or optionals, but output ",
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

int handle_negative_axis_validate(const std::string& attrib, int axis, int rank) {
  if (-rank > axis || axis >= rank) {
    fail_shape_inference(attrib, " axis value ", axis, " is invalid for a tensor of rank ", rank);
  }
  return (axis >= 0 ? axis : axis + rank);
}

void ScanInferenceFunction(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  auto num_scan_inputs = narrow<size_t>(getRequiredAttributeInt(ctx, "num_scan_inputs"));
  auto num_loop_state_vars = num_inputs - num_scan_inputs;
  auto num_outputs = ctx.getNumOutputs();
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
  subgraph_input_types.reserve(num_inputs);

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

      subgraph_input_types.push_back(input_type);
    } else {
      // For other inputs there is no fixed relationships to the Scan outputs,
      // so we don't propagate type/shape information.
      // We can pass through the type and shape to the subgraph inputs but
      // need to remove the sequence length dimensions from the shape.
      if (has_shape) {
        const auto& shape = input_type->tensor_type().shape();

        // remove sequence length dimensions and add to subgraph_input_types
        int axis = static_cast<int>(axes[i - num_loop_state_vars]);
        axis = handle_negative_axis_validate("scan_input_axes", axis, shape.dim_size());

        // update sequence_len if a value is available

        const auto& dims = shape.dim();
        mergeInDimensionInfo(dims.Get(axis), sequence_len_dim, 1);

        temporary_type_protos.emplace_back(RemoveIthDimensionFromShape(*input_type, axis));
        subgraph_input_types.emplace_back(&temporary_type_protos.back());

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
    input_data.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      // ctx.getInputData(i), the input to scan, does not represent the input to
      // scan body. So, we pass in null, to represent an unknown value.
      input_data.push_back(nullptr);
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
          output_axis = handle_negative_axis_validate("scan_output_axes", output_axis, output_rank);

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

void ScanVarLenInferenceFunction(InferenceContext& ctx) {
  // ScanVarLen has a single trailing variadic input at index 0, with layout
  //   [s_0..s_{N-1}, x_0..x_{M-1}, h_0..h_{K-1}]
  // where N = number of loop state variables (derived from the body subgraph),
  //       M = num_scan_inputs (attribute), and K = number of scan outputs.
  // The hint group [h_0..h_{K-1}] is either omitted entirely or present as
  // exactly K slots (with individual slots possibly empty placeholders).

  auto num_scan_inputs = narrow<size_t>(getRequiredAttributeInt(ctx, "num_scan_inputs"));
  if (num_scan_inputs < 1) {
    fail_shape_inference("ScanVarLen requires num_scan_inputs >= 1; got ", num_scan_inputs, ".");
  }

  // Derive N from the body subgraph (ground truth for the loop-state-var count).
  const auto* const body_attr = ctx.getAttribute("body");
  if (!body_attr || !body_attr->has_g()) {
    fail_type_inference("ScanVarLen requires a 'body' graph attribute.");
  }
  const GraphProto& body_graph = body_attr->g();
  const auto body_input_count = narrow<size_t>(body_graph.input_size());
  if (body_input_count < num_scan_inputs) {
    fail_shape_inference(
        "ScanVarLen body subgraph has ",
        body_input_count,
        " inputs; expected at least num_scan_inputs (",
        num_scan_inputs,
        ").");
  }
  const size_t num_loop_state_vars = body_input_count - num_scan_inputs; // N
  const size_t num_main_inputs = num_loop_state_vars + num_scan_inputs; // N + M

  // Derive K from the body subgraph as well (single source of truth: K is
  // body.output_count - N). Validate that the node's declared output count
  // matches N + K — a mismatch indicates a malformed node.
  const auto body_output_count = narrow<size_t>(body_graph.output_size());
  if (body_output_count < num_loop_state_vars) {
    fail_shape_inference(
        "ScanVarLen body subgraph has ",
        body_output_count,
        " output(s) but expected at least ",
        num_loop_state_vars,
        " (one per loop state variable, derived as body.input_size() - num_scan_inputs).");
  }
  const size_t num_scan_outputs = body_output_count - num_loop_state_vars; // K
  const size_t expected_num_outputs = num_loop_state_vars + num_scan_outputs;

  const auto num_outputs = ctx.getNumOutputs();
  if (num_outputs != expected_num_outputs) {
    fail_shape_inference(
        "ScanVarLen has ",
        num_outputs,
        " output(s) but the body subgraph implies ",
        expected_num_outputs,
        " (N = ",
        num_loop_state_vars,
        " state vars + K = ",
        num_scan_outputs,
        " scan outputs).");
  }

  // Validate hint arity: either 0 or exactly K hints.
  const auto num_total_inputs = ctx.getNumInputs();
  if (num_total_inputs < num_main_inputs) {
    fail_shape_inference(
        "ScanVarLen has ",
        num_total_inputs,
        " inputs; expected at least N + M = ",
        num_main_inputs,
        " (N = ",
        num_loop_state_vars,
        " derived from body subgraph, M = ",
        num_scan_inputs,
        ").");
  }
  const size_t num_hints = num_total_inputs - num_main_inputs;
  if (num_hints != 0 && num_hints != num_scan_outputs) {
    fail_shape_inference(
        "ScanVarLen hint count (",
        num_hints,
        ") must be either 0 or equal to the number of scan outputs K = ",
        num_scan_outputs,
        ".");
  }
  const bool has_hints = (num_hints == num_scan_outputs) && (num_scan_outputs > 0);

  // Per-hint static validation: each present hint must be a 1-D int64 tensor.
  // Length-vs-body-output-rank and non-concat-dim consistency are deferred
  // until after body inference (we need the body's output rank/shape).
  if (has_hints) {
    for (size_t i = 0; i < num_scan_outputs; ++i) {
      const size_t hint_idx = num_main_inputs + i;
      if (!ctx.hasInput(hint_idx)) {
        // Empty-string placeholder — this output has no hint.
        continue;
      }
      const auto* hint_type = ctx.getInputType(hint_idx);
      if (!hint_type || !hint_type->has_tensor_type() || hint_type->tensor_type().elem_type() != TensorProto::INT64) {
        fail_type_inference("ScanVarLen scan_output_shape_hint[", i, "] must be a tensor(int64).");
      }
      if (hint_type->tensor_type().has_shape()) {
        const auto& hint_shape = hint_type->tensor_type().shape();
        if (hint_shape.dim_size() != 1) {
          fail_shape_inference(
              "ScanVarLen scan_output_shape_hint[", i, "] must be a 1-D tensor; got rank ", hint_shape.dim_size(), ".");
        }
        const auto& length_dim = hint_shape.dim(0);
        if (length_dim.has_dim_value() && length_dim.dim_value() < 0) {
          fail_shape_inference(
              "ScanVarLen scan_output_shape_hint[", i, "] has invalid negative length ", length_dim.dim_value(), ".");
        }
      }
    }
  }

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

  // Build subgraph input types from the first N + M variadic inputs only.
  // Hints (when present) are at [N+M, N+M+K) and are NOT passed to the body.
  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_scan_inputs);

  std::vector<const TypeProto*> subgraph_input_types;
  subgraph_input_types.reserve(num_main_inputs);

  TensorShapeProto_Dimension sequence_len_dim;

  for (size_t i = 0; i < num_main_inputs; ++i) {
    const bool is_loop_state_var = i < num_loop_state_vars;
    const bool has_shape = hasInputShape(ctx, i);
    const auto* const input_type = ctx.getInputType(i);

    if (!input_type || !input_type->has_tensor_type()) {
      fail_type_inference("ScanVarLen input ", i, " was not a tensor.");
    }

    if (is_loop_state_var) {
      // Loop state variables propagate type and shape 1:1 from input to the
      // matching ScanVarLen output, and pass type/shape unchanged to the body subgraph.
      propagateElemTypeFromInputToOutput(ctx, i, i);
      if (has_shape) {
        propagateShapeFromInputToOutput(ctx, i, i);
      }
      subgraph_input_types.push_back(input_type);
    } else {
      // Scan inputs: strip the sequence axis before passing to the body subgraph
      // and merge the sequence length dim across all scan inputs.
      if (has_shape) {
        const auto& shape = input_type->tensor_type().shape();
        int axis = static_cast<int>(axes[i - num_loop_state_vars]);
        axis = handle_negative_axis_validate("scan_input_axes", axis, shape.dim_size());

        mergeInDimensionInfo(shape.dim(axis), sequence_len_dim, 1);

        temporary_type_protos.emplace_back(RemoveIthDimensionFromShape(*input_type, axis));
        subgraph_input_types.emplace_back(&temporary_type_protos.back());
      } else {
        subgraph_input_types.push_back(input_type);
      }
    }
  }

  // Zero-iteration is defined behavior in ScanVarLen v27 (Option B):
  // do NOT fail when sequence_len_dim is statically 0. Shape inference still
  // produces meaningful output shapes via the body subgraph (or via hints).

  // Run inferencing on the body subgraph.
  std::vector<const TypeProto*> output_types;
  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    // Scan inputs are sliced per-iteration, so we cannot pass their data to
    // the body. All entries are nullptr.
    std::vector<const TensorProto*> input_data(num_main_inputs, nullptr);
    output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // If empty(), assume inferencing was skipped.
  if (output_types.empty()) {
    return;
  }
  if (output_types.size() != num_outputs) {
    fail_type_inference(
        "Graph attribute inferencing returned type information for ",
        output_types.size(),
        " outputs. Expected ",
        num_outputs);
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    const bool is_loop_state_var = i < num_loop_state_vars;
    const auto* const subgraph_output_type = output_types[i];
    auto* op_output_type = ctx.getOutputType(i);
    auto* mutable_op_output_tensor_type = op_output_type->mutable_tensor_type();

    if (!subgraph_output_type->has_tensor_type()) {
      fail_type_inference("ScanVarLen 'body' subgraph outputs should all be tensors but output ", i, " was not.");
    }
    const auto& subgraph_output_tensor_type = subgraph_output_type->tensor_type();

    if (is_loop_state_var) {
      // Type has been propagated already; merge shape from the subgraph output.
      mergeInShapeInfo(subgraph_output_tensor_type, *mutable_op_output_tensor_type);
      continue;
    }

    // Scan output (i in [N, N+K)).
    mutable_op_output_tensor_type->set_elem_type(subgraph_output_tensor_type.elem_type());

    const size_t scan_out_idx = i - num_loop_state_vars;
    const size_t hint_idx = num_main_inputs + scan_out_idx;
    const bool hint_present = has_hints && ctx.hasInput(hint_idx);
    const TensorProto* hint_data = hint_present ? ctx.getInputData(hint_idx) : nullptr;

    const bool body_has_shape = subgraph_output_tensor_type.has_shape();
    int body_rank = body_has_shape ? subgraph_output_tensor_type.shape().dim_size() : -1;

    // C5: Validate declared hint length vs body output rank even when the
    // hint is non-constant. This catches mis-sized hints at shape inference
    // time rather than deferring the error to runtime — and applies whenever
    // the hint's declared shape provides a static dim_value for its length
    // (which is the common case, since hints are 1-D int64 tensors).
    if (hint_present && body_rank >= 0) {
      const auto* declared_hint_type = ctx.getInputType(hint_idx);
      if (declared_hint_type != nullptr && declared_hint_type->has_tensor_type() &&
          declared_hint_type->tensor_type().has_shape() && declared_hint_type->tensor_type().shape().dim_size() == 1) {
        const auto& length_dim = declared_hint_type->tensor_type().shape().dim(0);
        if (length_dim.has_dim_value() && length_dim.dim_value() != body_rank) {
          fail_shape_inference(
              "ScanVarLen scan_output_shape_hint[",
              scan_out_idx,
              "] has declared length ",
              length_dim.dim_value(),
              " but body subgraph output ",
              i,
              " has rank ",
              body_rank,
              ".");
        }
      }
    }

    if (hint_data != nullptr) {
      // CONSTANT HINT: derive a fully-static shape from the hint values.
      const std::vector<int64_t> hint_values = ParseData<int64_t>(hint_data);
      const auto hint_len = static_cast<int>(hint_values.size());

      // Validate hint length vs body output rank (if body rank is known).
      if (body_rank >= 0 && hint_len != body_rank) {
        fail_shape_inference(
            "ScanVarLen scan_output_shape_hint[",
            scan_out_idx,
            "] has length ",
            hint_len,
            " but body subgraph output ",
            i,
            " has rank ",
            body_rank,
            ".");
      }

      // Resolve the concat axis against the hint length (= output rank).
      int output_axis = static_cast<int>(output_axes[scan_out_idx]);
      output_axis = handle_negative_axis_validate("scan_output_axes", output_axis, hint_len);

      // Validate all hint values are non-negative.
      for (int j = 0; j < hint_len; ++j) {
        if (hint_values[j] < 0) {
          fail_shape_inference(
              "ScanVarLen scan_output_shape_hint[", scan_out_idx, "][", j, "] = ", hint_values[j], " is negative.");
        }
      }

      // Consistency check: non-concat hint dims must match body output's per-iteration dims.
      if (body_has_shape) {
        const auto& body_shape = subgraph_output_tensor_type.shape();
        for (int j = 0; j < hint_len; ++j) {
          if (j == output_axis) {
            continue; // concat axis — total may differ from per-iteration size
          }
          const auto& body_dim = body_shape.dim(j);
          if (body_dim.has_dim_value() && body_dim.dim_value() != hint_values[j]) {
            fail_shape_inference(
                "ScanVarLen scan_output_shape_hint[",
                scan_out_idx,
                "][",
                j,
                "] = ",
                hint_values[j],
                " disagrees with body subgraph output ",
                i,
                " dim ",
                j,
                " = ",
                body_dim.dim_value(),
                ". Non-concat hint dims must match the body output's per-iteration dims.");
          }
        }
      }

      // Strict-commitment check: when the iteration count is provably 0 (the
      // scan-input sequence axis has a static value of 0), the hint's concat-
      // axis value MUST also be 0. A non-zero value contradicts the actual
      // runtime output shape and is therefore a user error.
      if (sequence_len_dim.has_dim_value() && sequence_len_dim.dim_value() == 0 && hint_values[output_axis] != 0) {
        fail_shape_inference(
            "ScanVarLen scan_output_shape_hint[",
            scan_out_idx,
            "][",
            output_axis,
            "] = ",
            hint_values[output_axis],
            " at the concat axis, but the iteration count is provably 0 "
            "(producing an actual concat-axis size of 0). The hint is a "
            "strict commitment about the runtime output shape; mismatch is "
            "not allowed.");
      }

      // Emit fully-static output shape directly from the hint values.
      TensorShapeProto out_shape;
      for (int j = 0; j < hint_len; ++j) {
        out_shape.add_dim()->set_dim_value(hint_values[j]);
      }
      mergeInShapeInfo(out_shape, *mutable_op_output_tensor_type);
    } else if (body_has_shape) {
      // FALLBACK (no hint, "" placeholder, or non-constant hint):
      // use body shape with the concat-axis left symbolic.
      const TensorShapeProto& subgraph_output_shape = subgraph_output_tensor_type.shape();
      int output_axis = static_cast<int>(output_axes[scan_out_idx]);
      output_axis = handle_negative_axis_validate("scan_output_axes", output_axis, body_rank);

      TensorShapeProto inferred_shape;
      for (int j = 0; j < body_rank; ++j) {
        if (j == output_axis) {
          // Unknown dimension: total concat-axis size is data-dependent.
          inferred_shape.add_dim();
        } else {
          *inferred_shape.add_dim() = subgraph_output_shape.dim(j);
        }
      }
      mergeInShapeInfo(inferred_shape, *mutable_op_output_tensor_type);
    }
    // else: no body shape and no constant hint — nothing further to propagate.
  }
}

} // namespace ONNX_NAMESPACE
