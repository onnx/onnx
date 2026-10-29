// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapters for MeanVarianceNormalization between default-domain versions 28 and 29.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class MeanVarianceNormalizationAdapter : public Adapter {
 protected:
  MeanVarianceNormalizationAdapter(const OpSetID& initial_version, const OpSetID& target_version)
      : Adapter("MeanVarianceNormalization", initial_version, target_version) {}

  static Node* make_float_constant(const std::shared_ptr<Graph>& graph, Node* before, float value) {
    Tensor tensor;
    tensor.elem_type() = TensorProto_DataType_FLOAT;
    tensor.sizes() = {};
    tensor.floats() = {value};
    Node* constant = graph->create(kConstant);
    constant->t_(kvalue, tensor);
    constant->insertBefore(before);
    constant->output()->setElemType(TensorProto_DataType_FLOAT);
    return constant;
  }

  static Node* make_axes_constant(const std::shared_ptr<Graph>& graph, Node* before, const std::vector<int64_t>& axes) {
    Tensor tensor;
    tensor.elem_type() = TensorProto_DataType_INT64;
    tensor.sizes() = {static_cast<int64_t>(axes.size())};
    tensor.int64s() = axes;
    Node* constant = graph->create(kConstant);
    constant->t_(kvalue, tensor);
    constant->insertBefore(before);
    constant->output()->setElemType(TensorProto_DataType_INT64);
    return constant;
  }

  static Value*
  cast_to(const std::shared_ptr<Graph>& graph, Node* before, Value* input, TensorProto_DataType elem_type) {
    if (input->elemType() == elem_type) {
      return input;
    }
    Node* cast = graph->create(kCast);
    cast->i_(kto, elem_type);
    cast->addInput(input);
    cast->insertBefore(before);
    cast->output()->setElemType(elem_type);
    return cast->output();
  }

  static Node* make_binary(
      const std::shared_ptr<Graph>& graph,
      Node* before,
      Symbol op_type,
      Value* lhs,
      Value* rhs,
      TensorProto_DataType elem_type) {
    Node* result = graph->create(op_type);
    result->addInput(lhs);
    result->addInput(rhs);
    result->insertBefore(before);
    result->output()->setElemType(elem_type);
    return result;
  }

  static Node* make_reduce_mean(
      const std::shared_ptr<Graph>& graph,
      Node* before,
      Value* input,
      Value* axes,
      TensorProto_DataType elem_type) {
    Node* result = graph->create(kReduceMean);
    result->addInput(input);
    result->addInput(axes);
    result->insertBefore(before);
    result->output()->setElemType(elem_type);
    return result;
  }

  static Node* replace_node(Node* node, Node* replacement) {
    replacement->output()->setElemType(node->output()->elemType());
    if (!node->output()->sizes().empty()) {
      replacement->output()->setSizes(node->output()->sizes());
    }
    node->replaceAllUsesWith(replacement);
    node->destroy();
    return replacement;
  }
};

class MeanVarianceNormalization_28_29 final : public MeanVarianceNormalizationAdapter {
 public:
  MeanVarianceNormalization_28_29() : MeanVarianceNormalizationAdapter(OpSetID(28), OpSetID(29)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    ONNX_ASSERTM(node->inputs().size() == 1, "MeanVarianceNormalization must have exactly one input.")
    const auto input_type = static_cast<TensorProto_DataType>(node->inputs()[0]->elemType());
    ONNX_ASSERTM(
        input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_FLOAT ||
            input_type == TensorProto_DataType_DOUBLE || input_type == TensorProto_DataType_BFLOAT16,
        "MeanVarianceNormalization 28->29 upgrade requires a known floating-point input type, got ",
        input_type,
        ". Run shape inference before version conversion.")

    const std::vector<int64_t> axes = node->hasAttribute(kaxes) ? node->is(kaxes) : std::vector<int64_t>{0, 2, 3};
    Node* axes_constant = make_axes_constant(graph, node, axes);
    Value* exponent = cast_to(graph, node, make_float_constant(graph, node, 2.0f)->output(), input_type);
    Value* epsilon = cast_to(graph, node, make_float_constant(graph, node, 1e-9f)->output(), input_type);

    Node* mean = make_reduce_mean(graph, node, node->inputs()[0], axes_constant->output(), input_type);
    Node* mean_squared = make_binary(graph, node, kPow, mean->output(), exponent, input_type);
    Node* input_squared = make_binary(graph, node, kPow, node->inputs()[0], exponent, input_type);
    Node* squared_mean = make_reduce_mean(graph, node, input_squared->output(), axes_constant->output(), input_type);
    Node* variance = make_binary(graph, node, kSub, squared_mean->output(), mean_squared->output(), input_type);
    Node* stddev = graph->create(kSqrt);
    stddev->addInput(variance->output());
    stddev->insertBefore(node);
    stddev->output()->setElemType(input_type);
    Node* centered = make_binary(graph, node, kSub, node->inputs()[0], mean->output(), input_type);
    Node* denominator = make_binary(graph, node, kAdd, stddev->output(), epsilon, input_type);
    Node* result = make_binary(graph, node, kDiv, centered->output(), denominator->output(), input_type);
    return replace_node(node, result);
  }
};

class MeanVarianceNormalization_29_28 final : public MeanVarianceNormalizationAdapter {
 public:
  MeanVarianceNormalization_29_28() : MeanVarianceNormalizationAdapter(OpSetID(29), OpSetID(28)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    ONNX_ASSERTM(node->inputs().size() == 1, "MeanVarianceNormalization must have exactly one input.")
    const auto input_type = static_cast<TensorProto_DataType>(node->inputs()[0]->elemType());
    ONNX_ASSERTM(
        input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_FLOAT ||
            input_type == TensorProto_DataType_DOUBLE || input_type == TensorProto_DataType_BFLOAT16,
        "MeanVarianceNormalization 29->28 downgrade requires a known floating-point input type, got ",
        input_type,
        ". Run shape inference before version conversion.")
    const auto compute_type = input_type == TensorProto_DataType_FLOAT16 || input_type == TensorProto_DataType_BFLOAT16
        ? TensorProto_DataType_FLOAT
        : input_type;

    const std::vector<int64_t> axes = node->hasAttribute(kaxes) ? node->is(kaxes) : std::vector<int64_t>{0, 2, 3};
    const float epsilon_value = node->hasAttribute(kepsilon) ? node->f(kepsilon) : 1e-9f;
    Node* axes_constant = make_axes_constant(graph, node, axes);
    Value* input = cast_to(graph, node, node->inputs()[0], compute_type);
    Value* exponent = cast_to(graph, node, make_float_constant(graph, node, 2.0f)->output(), compute_type);
    Value* epsilon = cast_to(graph, node, make_float_constant(graph, node, epsilon_value)->output(), compute_type);

    Node* mean = make_reduce_mean(graph, node, input, axes_constant->output(), compute_type);
    Node* centered = make_binary(graph, node, kSub, input, mean->output(), compute_type);
    Node* centered_squared = make_binary(graph, node, kPow, centered->output(), exponent, compute_type);
    Node* variance = make_reduce_mean(graph, node, centered_squared->output(), axes_constant->output(), compute_type);
    Node* stddev = graph->create(kSqrt);
    stddev->addInput(variance->output());
    stddev->insertBefore(node);
    stddev->output()->setElemType(compute_type);
    Node* denominator = make_binary(graph, node, kAdd, stddev->output(), epsilon, compute_type);
    Node* normalized = make_binary(graph, node, kDiv, centered->output(), denominator->output(), compute_type);
    Value* result = cast_to(graph, node, normalized->output(), input_type);
    return replace_node(node, result->node());
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
