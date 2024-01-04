// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for GroupNormalization in default domain from version 20 to 21

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class GroupNormalization_20_21 final : public Adapter {
 public:
  explicit GroupNormalization_20_21() : Adapter("GroupNormalization", OpSetID(20), OpSetID(21)) {}

  void transform_input(std::shared_ptr<Graph> graph, Node* node, int64_t input_id) const {
    // Perform following sequence of ops on input, effect is similar to numpy.repeat()

    //   -- Shape<start=1,end=2>() -- Div(Shape_out, num_groups)
    //  /                                             |
    // --  Reshape(,[-1,1]) ------------- Expand(Reshape_out,[1,Div_out]) -- Reshape(Expand_out, [-1])

    Symbol kShape("Shape");
    Node* shape = graph->create(kShape);
    shape->i_(kstart, 1);
    shape->i_(kend, 2);
    shape->addInput(node->inputs()[input_id]);
    shape->insertBefore(node);

    Tensor tensor_num_groups;
    tensor_num_groups.elem_type() = TensorProto_DataType_INT64;
    int64_t num_groups = node->i(knum_groups);
    tensor_num_groups.sizes() = {1};
    tensor_num_groups.int64s() = {num_groups};
    Node* constant_num_grp = graph->create(kConstant);
    constant_num_grp->t_(kvalue, tensor_num_groups);
    constant_num_grp->insertBefore(node);

    Node* div = graph->create(kDiv);
    div->addInput(shape->output());
    div->addInput(constant_num_grp->output());
    div->insertBefore(node);

    Tensor tensor_reshape0_shape;
    tensor_reshape0_shape.elem_type() = TensorProto_DataType_INT64;
    tensor_reshape0_shape.sizes() = {2};
    tensor_reshape0_shape.int64s() = {-1, 1};
    Node* constant_reshape0_shape = graph->create(kConstant);
    constant_reshape0_shape->t_(kvalue, tensor_reshape0_shape);
    constant_reshape0_shape->insertBefore(node);
    Node* reshape0 = graph->create(kReshape);
    reshape0->addInput(node->inputs()[input_id]);
    reshape0->addInput(constant_reshape0_shape->output());
    reshape0->insertBefore(node);

    Tensor tensor_one;
    tensor_one.elem_type() = TensorProto_DataType_INT64;
    tensor_one.sizes() = {1};
    tensor_one.int64s() = {1};
    Node* constant_one = graph->create(kConstant);
    constant_one->t_(kvalue, tensor_one);
    constant_one->insertBefore(node);
    Node* concat = graph->create(kConcat);
    concat->i_(kaxis, 0);
    concat->addInput(constant_one->output());
    concat->addInput(div->output());
    concat->insertBefore(node);
    Node* expand = graph->create(kExpand);
    expand->addInput(reshape0->output());
    expand->addInput(concat->output());
    expand->insertBefore(node);

    Tensor tensor_reshape1_shape;
    tensor_reshape1_shape.elem_type() = TensorProto_DataType_INT64;
    tensor_reshape1_shape.sizes() = {1};
    tensor_reshape1_shape.int64s() = {-1};
    Node* constant_reshape1_shape = graph->create(kConstant);
    constant_reshape1_shape->t_(kvalue, tensor_reshape1_shape);
    constant_reshape1_shape->insertBefore(node);
    Node* reshape1 = graph->create(kReshape);
    reshape1->addInput(expand->output());
    reshape1->addInput(constant_reshape1_shape->output());
    reshape1->insertBefore(node);

    node->replaceInput(input_id, reshape1->output());
  }

  void adapt_group_normalization_20_21(std::shared_ptr<Graph> graph, Node* node) const {
    transform_input(graph, node, 1);
    transform_input(graph, node, 2);
    node->i_(kstash_type, node->inputs()[0]->elemType());
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_group_normalization_20_21(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
