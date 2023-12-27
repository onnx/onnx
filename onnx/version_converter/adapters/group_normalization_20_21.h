// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for GroupNormalization in default domain from version 20 to 21

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class GroupNormalization_20_21 final : public Adapter {
 public:
  explicit GroupNormalization_20_21() : Adapter("GroupNormalization", OpSetID(20), OpSetID(21)) {}

  void transform_input(std::shared_ptr<Graph> graph, Node* node, int64_t input_id) const {
  /*
  Perform following sequence of ops on input, effect is similar to numpy.repeat()

      -- Shape<start=1,end=2>() -- Div(Shape_out, num_groups) 
    /                                             \
  --  Reshape(,[-1,1]) ------------- Expand(Reshape_out,[1,Div_out]) -- Reshape(Expand_out, [-1])
  */
    Symbol kShape("Shape");
    Node* shape = graph->create(kShape);
    shape->i_(kstart, 1);
    shape->i_(kend, 2);
    shape->addInput(node->inputs()[input_id]);
    shape->insertBefore(node);

    Tensor t0;
    t0.elem_type() = TensorProto_DataType_INT64;
    int64_t num_groups = node->i(knum_groups);
    t0.sizes() = {1};
    t0.int64s() = {num_groups};
    Node* constant0 = graph->create(kConstant);
    constant0->t_(kvalue, t0);
    constant0->insertBefore(node);

    Node* div = graph->create(kDiv);
    div->addInput(shape->output());
    div->addInput(constant0->output());
    div->insertBefore(node);

    Tensor t1;
    t1.elem_type() = TensorProto_DataType_INT64;
    t1.sizes() = {2};
    t1.int64s() = {-1, 1};
    Node* constant1 = graph->create(kConstant);
    constant1->t_(kvalue, t1);
    constant1->insertBefore(node);
    Node* reshape0 = graph->create(kReshape);
    reshape0->addInput(node->inputs()[1]);
    reshape0->addInput(constant1->output());
    reshape0->insertBefore(node);

    Tensor t2;
    t2.elem_type() = TensorProto_DataType_INT64;
    t2.sizes() = {1};
    t2.int64s() = {1};
    Node* constant2 = graph->create(kConstant);
    constant2->t_(kvalue, t2);
    constant2->insertBefore(node);
    Node* concat = graph->create(kConcat);
    concat->i_(kaxis, 0);
    concat->addInput(constant2->output());
    concat->addInput(div->output());
    concat->insertBefore(node);
    Node* expand = graph->create(kExpand);
    expand->addInput(reshape0->output());
    expand->addInput(concat->output());
    expand->insertBefore(node);

    Tensor t3;
    t3.elem_type() = TensorProto_DataType_INT64;
    t3.sizes() = {1};
    t3.int64s() = {-1};
    Node* constant3 = graph->create(kConstant);
    constant3->t_(kvalue, t3);
    constant3->insertBefore(node);
    Node* reshape1 = graph->create(kReshape);
    reshape1->addInput(expand->output());
    reshape1->addInput(constant3->output());
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

