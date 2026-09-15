// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for GroupNormalization in default domain from version 21 to 20

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class GroupNormalization_21_20 final : public Adapter {
 public:
  explicit GroupNormalization_21_20() : Adapter("GroupNormalization", OpSetID(21), OpSetID(20)) {}

  Node* make_int64_constant(const std::shared_ptr<Graph>& graph, Node* before, const std::vector<int64_t>& values)
      const {
    Tensor tensor;
    tensor.elem_type() = TensorProto_DataType_INT64;
    tensor.sizes() = {static_cast<int64_t>(values.size())};
    tensor.int64s() = values;
    Node* constant = graph->create(kConstant);
    constant->t_(kvalue, tensor);
    constant->insertBefore(before);
    return constant;
  }

  Node* make_float_constant(const std::shared_ptr<Graph>& graph, Node* before, float value) const {
    Tensor tensor;
    tensor.elem_type() = TensorProto_DataType_FLOAT;
    tensor.sizes() = {};
    tensor.floats() = {value};
    Node* constant = graph->create(kConstant);
    constant->t_(kvalue, tensor);
    constant->insertBefore(before);
    return constant;
  }

  Value* cast_to(const std::shared_ptr<Graph>& graph, Node* before, Value* input, TensorProto_DataType type) const {
    if (input->elemType() == type) {
      return input;
    }
    Node* cast = graph->create(kCast);
    cast->i_(kto, type);
    cast->addInput(input);
    cast->insertBefore(before);
    cast->output()->setElemType(type);
    return cast->output();
  }

  Node* adapt_group_normalization_21_20(const std::shared_ptr<Graph>& graph, Node* node) const {
    ONNX_ASSERTM(node->inputs().size() == 3, "GroupNormalization node must have exactly 3 inputs")
    ONNX_ASSERTM(node->outputs().size() == 1, "GroupNormalization node must have exactly 1 output")

    // Preserve the input and accumulation types used by the opset-21 operator.
    const int64_t input_type = node->inputs()[0]->elemType();
    ONNX_ASSERTM(
        input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_FLOAT ||
            input_type == TensorProto_DataType_DOUBLE || input_type == TensorProto_DataType_BFLOAT16,
        "GroupNormalization input type ",
        input_type,
        " is not supported when converting from opset 21 to opset 20.")

    const int64_t stash_type =
        node->hasAttribute(kstash_type) ? node->i(kstash_type) : static_cast<int64_t>(TensorProto_DataType_FLOAT);
    ONNX_ASSERTM(
        stash_type == TensorProto_DataType_FLOAT16 || stash_type == TensorProto_DataType_FLOAT ||
            stash_type == TensorProto_DataType_DOUBLE,
        "GroupNormalization stash_type ",
        stash_type,
        " cannot be represented with InstanceNormalization in opset 20.")

    const int64_t num_groups = node->i(knum_groups);
    ONNX_ASSERTM(num_groups > 0, "GroupNormalization num_groups must be greater than zero.")
    const float epsilon = node->hasAttribute(kepsilon) ? static_cast<float>(node->f(kepsilon)) : 1e-5f;

    Value* x = node->inputs()[0];
    Value* scale = node->inputs()[1];
    Value* bias = node->inputs()[2];
    Value* x_stashed = cast_to(graph, node, x, static_cast<TensorProto_DataType>(stash_type));

    // Collapse each group to one InstanceNormalization channel.
    Node* grouped_shape = make_int64_constant(graph, node, {0, num_groups, -1});
    Node* grouped_x = graph->create(kReshape);
    grouped_x->addInput(x_stashed);
    grouped_x->addInput(grouped_shape->output());
    grouped_x->insertBefore(node);

    Node* group_shape = make_int64_constant(graph, node, {num_groups});
    Node* one = make_float_constant(graph, node, 1.0f);
    Value* one_stashed = cast_to(graph, node, one->output(), static_cast<TensorProto_DataType>(stash_type));
    Node* instance_scale = graph->create(kExpand);
    instance_scale->addInput(one_stashed);
    instance_scale->addInput(group_shape->output());
    instance_scale->insertBefore(node);

    Node* zero = make_float_constant(graph, node, 0.0f);
    Value* zero_stashed = cast_to(graph, node, zero->output(), static_cast<TensorProto_DataType>(stash_type));
    Node* instance_bias = graph->create(kExpand);
    instance_bias->addInput(zero_stashed);
    instance_bias->addInput(group_shape->output());
    instance_bias->insertBefore(node);

    const Symbol kInstanceNormalization("InstanceNormalization");
    Node* normalized_grouped = graph->create(kInstanceNormalization);
    normalized_grouped->f_(kepsilon, epsilon);
    normalized_grouped->addInput(grouped_x->output());
    normalized_grouped->addInput(instance_scale->output());
    normalized_grouped->addInput(instance_bias->output());
    normalized_grouped->insertBefore(node);

    // Restore the original shape and input type before applying channel-wise affine parameters.
    const Symbol kShape("Shape");
    Node* x_shape = graph->create(kShape);
    x_shape->addInput(x);
    x_shape->insertBefore(node);
    Node* normalized_shape = graph->create(kReshape);
    normalized_shape->addInput(normalized_grouped->output());
    normalized_shape->addInput(x_shape->output());
    normalized_shape->insertBefore(node);
    Value* normalized = cast_to(graph, node, normalized_shape->output(), static_cast<TensorProto_DataType>(input_type));

    // Build [1, C, 1, ...] without requiring static rank or dimensions.
    Node* shape_of_shape = graph->create(kShape);
    shape_of_shape->addInput(x_shape->output());
    shape_of_shape->insertBefore(node);
    Node* two = make_int64_constant(graph, node, {2});
    Node* suffix_length = graph->create(kSub);
    suffix_length->addInput(shape_of_shape->output());
    suffix_length->addInput(two->output());
    suffix_length->insertBefore(node);

    Tensor tensor_one;
    tensor_one.elem_type() = TensorProto_DataType_INT64;
    tensor_one.sizes() = {1};
    tensor_one.int64s() = {1};
    const Symbol kConstantOfShape("ConstantOfShape");
    Node* suffix_ones = graph->create(kConstantOfShape);
    suffix_ones->t_(kvalue, tensor_one);
    suffix_ones->addInput(suffix_length->output());
    suffix_ones->insertBefore(node);

    Node* affine_prefix = make_int64_constant(graph, node, {1, -1});
    Node* affine_shape = graph->create(kConcat);
    affine_shape->i_(kaxis, 0);
    affine_shape->addInput(affine_prefix->output());
    affine_shape->addInput(suffix_ones->output());
    affine_shape->insertBefore(node);

    Node* scale_reshaped = graph->create(kReshape);
    scale_reshaped->addInput(scale);
    scale_reshaped->addInput(affine_shape->output());
    scale_reshaped->insertBefore(node);
    Node* bias_reshaped = graph->create(kReshape);
    bias_reshaped->addInput(bias);
    bias_reshaped->addInput(affine_shape->output());
    bias_reshaped->insertBefore(node);

    Node* scaled = graph->create(kMul);
    scaled->addInput(normalized);
    scaled->addInput(scale_reshaped->output());
    scaled->insertBefore(node);
    Node* result = graph->create(kAdd);
    result->addInput(scaled->output());
    result->addInput(bias_reshaped->output());
    result->insertBefore(node);

    node->replaceAllUsesWith(result);
    node->destroy();
    return result;
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    return adapt_group_normalization_21_20(graph, node);
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
