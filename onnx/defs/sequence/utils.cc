// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include "onnx/defs/sequence/utils.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::defs::sequence::utils {

void SplitToSequenceShapeInference(InferenceContext& ctx) {
  const auto* const input0_type = ctx.getInputType(0);
  if (nullptr == input0_type) {
    fail_type_inference("Input type for input at index 0 is null. Type info is expected.")
  }
  ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
      input0_type->tensor_type().elem_type());

  if (!hasInputShape(ctx, 0)) {
    return;
  }

  const auto& inputShape = input0_type->tensor_type().shape();

  int r = inputShape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -r || axis > r - 1) {
    fail_shape_inference("Invalid value of attribute 'axis'. Rank=", r, " Value=", axis);
  }
  if (axis < 0) {
    axis += r;
  }

  int64_t splitSize = 1;
  int64_t keepdims = 1;
  if (!ctx.hasInput(1)) {
    // input split is omitted, default to split by 1.
    const auto* const attr_proto = ctx.getAttribute("keepdims");
    if (attr_proto) {
      keepdims = attr_proto->i();
    }
  } else {
    splitSize = [&]() -> int64_t {
      // Need input split shape info and initializer data to infer split sizes.
      if (!hasInputShape(ctx, 1)) {
        return -1;
      }
      const TensorProto* splitInitializer = ctx.getInputData(1);
      if (nullptr == splitInitializer || !splitInitializer->has_data_type()) {
        return -1;
      }

      std::vector<int64_t> splitSizes;
      if (splitInitializer->data_type() == TensorProto::INT64) {
        const auto data = ParseData<int64_t>(splitInitializer);
        splitSizes.insert(splitSizes.end(), data.begin(), data.end());
      } else if (splitInitializer->data_type() == TensorProto::INT32) {
        const auto data = ParseData<int32_t>(splitInitializer);
        splitSizes.insert(splitSizes.end(), data.begin(), data.end());
      } else {
        // unaccepted data type
        fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for split");
      }

      if (splitSizes.empty()) {
        fail_shape_inference("Input 'split' can not be empty.");
      }

      const auto& splitDim = inputShape.dim(axis);
      if (!splitDim.has_dim_value()) {
        // Unable to verify nor infer exact split dimension size.
        return -1;
      }

      int64_t splitDimValue = splitDim.dim_value();
      const auto& splitShape = getInputShape(ctx, 1);
      if (splitShape.dim_size() == 0) {
        // split is scalar
        if (splitSizes[0] <= 0) {
          fail_shape_inference("Scalar input 'split' must be greater than zero.");
        }
        if (splitDimValue % splitSizes[0] == 0) {
          // all output chunks have the same shape, assign that to output sequence shape.
          return splitSizes[0];
        }
        return -1;
      } else {
        // split is 1-D tensor
        int64_t splitSizesSum = 0;
        for (const int64_t splitSize : splitSizes) {
          if (checked_add_overflow(splitSizesSum, splitSize, &splitSizesSum)) {
            fail_shape_inference("Sum of split values is too large.");
          }
        }
        if (splitDimValue != splitSizesSum) {
          fail_shape_inference(
              "Sum of split values not equal to 'input' dim size on 'axis'. 'axis' dim size=",
              splitDimValue,
              " sum of split values=",
              splitSizesSum);
        }
        if (std::adjacent_find(splitSizes.begin(), splitSizes.end(), std::not_equal_to()) == splitSizes.end()) {
          // all split sizes are the same.
          return splitSizes[0];
        }
        return -1;
      }
    }();
  }

  if (keepdims) {
    auto* outputShape =
        ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->mutable_shape();
    *outputShape = inputShape;
    auto* dim = outputShape->mutable_dim(axis);
    // Tensors in sequence could not have different shapes explicitly.
    // Only assign dim_value when all chunks have the same shape.
    if (splitSize > 0) {
      dim->set_dim_value(splitSize);
    } else {
      dim->clear_dim_value();
      dim->clear_dim_param();
    }
  } else {
    TensorShapeProto* outputShape =
        ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < inputShape.dim_size(); ++i) {
      if (i != axis) {
        auto* dim = outputShape->add_dim();
        dim->CopyFrom(inputShape.dim(i));
      }
    }
  }
}

std::function<void(OpSchema&)> SplitToSequenceOpGenerator() {
  return [](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(SplitToSequenceShapeInference); };
}

void SequenceEmptyShapeInference_opset11(InferenceContext& ctx) {
  const auto attr_proto = ctx.getAttribute("dtype");
  auto elem_type = TensorProto::FLOAT;
  if (nullptr != attr_proto) {
    if (!attr_proto->has_i()) {
      fail_type_inference("Attribute dtype should be of integer type and specify a type.");
    }
    auto attr_value = attr_proto->i();
    elem_type = static_cast<TensorProto_DataType>(attr_value);
  }
  ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(elem_type);
}

void SequenceConstructShapeInference_opset11(InferenceContext& ctx) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs < 1) {
    fail_type_inference("SequenceConstruct is expected to have at least 1 input.");
  }

  std::vector<int> input_elem_types;
  input_elem_types.reserve(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    auto input_type = ctx.getInputType(i);
    if (nullptr == input_type) {
      fail_type_inference("Input type for input at index ", i, " is null. Type info is expected.");
    }
    input_elem_types.emplace_back(input_type->tensor_type().elem_type());
  }
  if (std::adjacent_find(input_elem_types.begin(), input_elem_types.end(), std::not_equal_to()) !=
      input_elem_types.end()) {
    // not all input elem types are the same.
    fail_type_inference("Element type of inputs are expected to be the same.");
  }

  auto output_tensor_type = ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();

  output_tensor_type->set_elem_type(static_cast<TensorProto_DataType>(input_elem_types[0]));

  if (!hasNInputShapes(ctx, numInputs)) {
    return;
  }

  *(output_tensor_type->mutable_shape()) = ctx.getInputType(0)->tensor_type().shape();

  for (size_t i = 1; i < numInputs; ++i) {
    const auto& input_shape = ctx.getInputType(i)->tensor_type().shape();
    UnionShapeInfo(input_shape, *output_tensor_type);
  }
}

void SequenceInsertShapeInference_opset11(InferenceContext& ctx) {
  const auto input0_type = ctx.getInputType(0);
  const auto input1_type = ctx.getInputType(1);
  if (nullptr == input0_type || nullptr == input1_type) {
    fail_type_inference("Input Sequence and Tensor are expected to have type info. Current type is null.");
  }
  const auto seq_elem_type = input0_type->sequence_type().elem_type().tensor_type().elem_type();
  const auto tensor_elem_type = input1_type->tensor_type().elem_type();
  if (seq_elem_type != tensor_elem_type) {
    fail_type_inference(
        "Input Sequence and Tensor are expected to have the same elem type. Sequence=",
        seq_elem_type,
        " Tensor=",
        tensor_elem_type);
  }

  auto output_tensor_type = ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  output_tensor_type->set_elem_type(seq_elem_type);

  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  *(output_tensor_type->mutable_shape()) = input0_type->sequence_type().elem_type().tensor_type().shape();

  UnionShapeInfo(input1_type->tensor_type().shape(), *output_tensor_type);
}

void SequenceAtShapeInference_opset11(InferenceContext& ctx) {
  const auto input0_type = ctx.getInputType(0);
  if (nullptr == input0_type) {
    fail_type_inference("Input type for input at index 0 is null. Type info is expected.");
  }
  ctx.getOutputType(0)->CopyFrom(input0_type->sequence_type().elem_type());
}

void SequenceEraseShapeInference_opset11(InferenceContext& ctx) {
  const auto input0_type = ctx.getInputType(0);
  if (nullptr == input0_type) {
    fail_type_inference("Input type for input at index 0 is null. Type info is expected.")
  }
  ctx.getOutputType(0)->CopyFrom(*input0_type);
}

void SequenceLengthShapeInference_opset11(InferenceContext& ctx) {
  auto output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_tensor_type->set_elem_type(TensorProto::INT64);
  output_tensor_type->mutable_shape()->Clear();
}

void ConcatFromSequenceShapeInference_opset11(InferenceContext& ctx) {
  const auto input0_type = ctx.getInputType(0);
  if (nullptr == input0_type) {
    fail_type_inference("Input type for input at index 0 is null. Type info is expected.")
  }
  auto elem_type = input0_type->sequence_type().elem_type().tensor_type().elem_type();
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(elem_type);

  if (!hasInputShape(ctx, 0)) {
    return;
  }

  auto axis_attr = ctx.getAttribute("axis");
  if (!axis_attr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axis_attr->i());

  int new_axis = 0;
  auto new_axis_attr = ctx.getAttribute("new_axis");
  if (new_axis_attr) {
    new_axis = static_cast<int>(new_axis_attr->i());
  }

  const auto& input_shape = ctx.getInputType(0)->sequence_type().elem_type().tensor_type().shape();
  auto rank = input_shape.dim_size();
  if (1 != new_axis && 0 != new_axis) {
    fail_shape_inference("new_axis must be either 0 or 1");
  }

  auto upper_bound = 1 == new_axis ? rank : rank - 1;
  auto lower_bound = 1 == new_axis ? -rank - 1 : -rank;

  if (axis < lower_bound || axis > upper_bound) {
    fail_shape_inference(
        "Invalid value of attribute 'axis'. Accepted range=[", lower_bound, ", ", upper_bound, "], Value=", axis);
  }

  if (axis < 0) {
    axis += (upper_bound + 1);
  }

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i <= upper_bound; ++i) {
    output_shape->add_dim();
    if (i != axis) {
      output_shape->mutable_dim(i)->CopyFrom(input_shape.dim((i > axis && new_axis) ? i - 1 : i));
    }
  }
}

void SequenceMapInferenceFunction(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  assert(num_inputs > 0);

  auto num_outputs = ctx.getNumOutputs();
  assert(num_outputs > 0);

  std::vector<TypeProto> tmp_type_protos(num_inputs);
  std::vector<const TypeProto*> subgraph_input_types;
  subgraph_input_types.reserve(num_inputs);
  for (size_t inputIndex = 0; inputIndex < num_inputs; inputIndex++) {
    const auto* const input_type = ctx.getInputType(inputIndex);
    if (input_type == nullptr) {
      fail_type_inference("Input ", inputIndex, " expected to have type info");
    }
    if (input_type->value_case() == TypeProto::kSequenceType) {
      tmp_type_protos[inputIndex].CopyFrom(input_type->sequence_type().elem_type());
      subgraph_input_types.push_back(&tmp_type_protos[inputIndex]);
    } else {
      if (inputIndex == 0)
        fail_type_inference("Input ", inputIndex, " expected to be a sequence type");
      subgraph_input_types.push_back(input_type);
    }
  }

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (!graphInferencer)
    fail_type_inference("Graph attribute inferencer for \"body\" not available");

  std::vector<const TensorProto*> input_data(num_inputs, nullptr);
  std::vector<const TypeProto*> subgraph_output_types =
      graphInferencer->doInferencing(subgraph_input_types, input_data);

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    if (subgraph_output_types.size() != num_outputs) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs);
    }

    for (size_t outputIndex = 0; outputIndex < num_outputs; outputIndex++) {
      const auto* const subgraph_output_type = subgraph_output_types[outputIndex];
      ctx.getOutputType(outputIndex)->mutable_sequence_type()->mutable_elem_type()->CopyFrom(*subgraph_output_type);
    }
  }
}

bool BuildSequenceMapBodyFunc(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  schema.BuildFunction(functionProto);

  // variadic input/outputs will be expanded
  functionProto.clear_input();
  functionProto.clear_output();

  const auto* const body_attr = ctx.getAttribute("body");
  if (!body_attr || !body_attr->has_g())
    ONNX_THROW_EX(std::invalid_argument("Invalid ``body`` argument. Expected a graph"));
  const GraphProto& body = body_attr->g();

  auto g_inputs = body.input();
  int ninputs = g_inputs.size();
  if (ninputs < 1)
    ONNX_THROW_EX(std::invalid_argument("Expected 1 or more inputs."));

  auto g_outputs = body.output();
  int noutputs = g_outputs.size();
  if (noutputs < 1)
    ONNX_THROW_EX(std::invalid_argument("Expected 1 or more outputs."));

  if (!ctx.hasInput(0))
    ONNX_THROW_EX(std::invalid_argument("Input 0 expected but not provided"));

  const auto* const first_input_type = ctx.getInputType(0);
  // A serialized input may be present without type metadata.
  if (first_input_type == nullptr || !first_input_type->has_sequence_type())
    ONNX_THROW_EX(std::invalid_argument("Expected a sequence type for input 0"));

  auto const& schema_inputs = schema.inputs();
  auto const& input_0_name = schema_inputs[0].GetName();
  auto const& input_1_name = schema_inputs[1].GetName(); // variadic input

  *functionProto.add_input() = input_0_name;
  for (int i = 1; i < ninputs; i++) {
    if (!ctx.hasInput(i))
      ONNX_THROW_EX(std::invalid_argument(MakeString("Input ", i, " expected but not provided")));
    *functionProto.add_input() = MakeString(input_1_name, "_", i);
  }

  auto const& schema_outputs = schema.outputs();
  auto output_0_name = schema_outputs[0].GetName();
  for (int i = 0; i < noutputs; i++) {
    if (!ctx.hasOutput(i))
      ONNX_THROW_EX(std::invalid_argument(MakeString("Output ", i, " expected but not provided")));
    *functionProto.add_output() = MakeString(output_0_name, "_", i);
  }

  // Loop body subgraph
  const std::string loopbody_graph_name("SequenceMap_loop_body");
  GraphProto loopbody_graph;
  loopbody_graph.set_name(loopbody_graph_name);
  {
    TypeProto int64_type;
    int64_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_type.mutable_tensor_type()->mutable_shape()->Clear();

    TypeProto bool_type;
    bool_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_type.mutable_tensor_type()->mutable_shape()->Clear();

    ValueInfoProto iter_count;
    std::string iter_count_name = loopbody_graph_name + "_itercount";
    iter_count.set_name(iter_count_name);
    *iter_count.mutable_type() = int64_type;
    *loopbody_graph.add_input() = iter_count;

    ValueInfoProto cond_in;
    std::string cond_in_name = loopbody_graph_name + "_cond_in";
    cond_in.set_name(cond_in_name);
    *cond_in.mutable_type() = bool_type;
    *loopbody_graph.add_input() = cond_in;

    ValueInfoProto cond_out;
    std::string cond_out_name = loopbody_graph_name + "_cond_out";
    cond_out.set_name(cond_out_name);
    *cond_out.mutable_type() = bool_type;
    *loopbody_graph.add_output() = cond_out;

    NodeProto cond_identity;
    cond_identity.set_domain(ONNX_DOMAIN);
    cond_identity.set_op_type("Identity");
    cond_identity.add_input(cond_in_name);
    cond_identity.add_output(cond_out_name);
    *loopbody_graph.add_node() = cond_identity;

    for (int inputIndex = 0; inputIndex < ninputs; inputIndex++) {
      const auto* const input_type = ctx.getInputType(inputIndex);
      if (input_type && input_type->has_sequence_type()) {
        // If it's a sequence input, extract ``iter_count`` element
        NodeProto seq_at_node;
        seq_at_node.set_domain(ONNX_DOMAIN);
        seq_at_node.set_op_type("SequenceAt");
        seq_at_node.add_input(functionProto.input(inputIndex));
        seq_at_node.add_input(iter_count_name);
        seq_at_node.add_output(g_inputs.Get(inputIndex).name());
        *loopbody_graph.add_node() = std::move(seq_at_node);
      } else {
        // If not a sequence, simply connect
        NodeProto identity;
        identity.set_domain(ONNX_DOMAIN);
        identity.set_op_type("Identity");
        identity.add_input(functionProto.input(inputIndex));
        identity.add_output(g_inputs.Get(inputIndex).name());
        *loopbody_graph.add_node() = std::move(identity);
      }
    }

    for (const auto& item : body.node())
      *loopbody_graph.add_node() = item;
    for (const auto& item : body.value_info())
      *loopbody_graph.add_value_info() = item;
    for (const auto& item : body.initializer())
      *loopbody_graph.add_initializer() = item;
    for (const auto& item : body.sparse_initializer())
      *loopbody_graph.add_sparse_initializer() = item;

    for (int outputIndex = 0; outputIndex < noutputs; outputIndex++) {
      const auto& body_out_i = body.output(outputIndex);
      assert(body_out_i.type().has_tensor_type());
      std::string prefix = loopbody_graph_name + "_" + body_out_i.name();
      std::string loopbody_in_name = prefix + "_in";

      ValueInfoProto tmp;
      *tmp.mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type() =
          body_out_i.type().tensor_type();
      tmp.set_name(loopbody_in_name);
      *loopbody_graph.add_input() = tmp;

      std::string loopbody_out_name = prefix + "_out";
      tmp.set_name(loopbody_out_name);
      *loopbody_graph.add_output() = tmp;

      NodeProto seq_insert_node;
      seq_insert_node.set_domain(ONNX_DOMAIN);
      seq_insert_node.set_op_type("SequenceInsert");
      seq_insert_node.add_input(loopbody_in_name);
      seq_insert_node.add_input(body_out_i.name());
      seq_insert_node.add_output(loopbody_out_name);
      *loopbody_graph.add_node() = seq_insert_node;
    }
  }

  std::vector<FunctionBodyHelper::NodeDef> nodes;

  // TODO(ONNX): figure out a way to prevent name collisions?
  auto first_input_name = functionProto.input(0);
  std::string prefix = MakeString("SequenceMap_", first_input_name);
  std::string seqlen = prefix + "_seqlen";
  nodes.push_back({{seqlen}, "SequenceLength", {first_input_name}});

  std::string cond_bool = prefix + "_cond";
  nodes.push_back(FunctionBodyHelper::Const<bool>(cond_bool, true));

  std::vector<std::string> loop_node_inputs = {seqlen, cond_bool};
  std::vector<std::string> loop_node_outputs;
  for (int outputIndex = 0; outputIndex < noutputs; outputIndex++) {
    auto output_name = functionProto.output(outputIndex);
    std::string out_prefix = MakeString("SequenceMap_", output_name);

    std::string seqempty_name = out_prefix + "_seqempty";
    int64_t dtype = g_outputs.Get(outputIndex).type().tensor_type().elem_type();
    nodes.push_back({{seqempty_name}, "SequenceEmpty", {}, {MakeAttribute("dtype", dtype)}});
    loop_node_inputs.emplace_back(std::move(seqempty_name));
    loop_node_outputs.emplace_back(std::move(output_name));
  }

  nodes.push_back({loop_node_outputs, "Loop", loop_node_inputs, {MakeAttribute("body", loopbody_graph)}});

  auto func_nodes = FunctionBodyHelper::BuildNodes(nodes);
  for (const auto& node : func_nodes) {
    auto* new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  return true;
}

} // namespace ONNX_NAMESPACE::defs::sequence::utils
