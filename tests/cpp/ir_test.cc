// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/common/assertions.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE::Test {

static bool IsValidIdentifier(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  if (!IsAlpha(name[0]) && name[0] != '_') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!IsAlnum(name[i]) && name[i] != '_') {
      return false;
    }
  }
  return true;
}

TEST(IR, ValidIdentifierTest) {
  const std::string name = "initializer_name_longer_than_small_string_storage";
  const std::string raw_data(4096, '\x01');
  for (int64_t ir_version : {3, 8}) {
    for (bool is_input : {false, true}) {
      if (ir_version < 4 && !is_input) {
        continue;
      }
      SCOPED_TRACE(::testing::Message() << "ir_version=" << ir_version << ", is_input=" << is_input);
      auto g = std::make_shared<Graph>();
      g->setName("test");
      Tensor initializer;
      initializer.setName(name);
      initializer.elem_type() = TensorProto_DataType_FLOAT;
      initializer.sizes() = {1024};
      initializer.set_raw_data(raw_data);
      Value* x = nullptr;
      if (is_input) {
        x = g->addInput();
        x->setUniqueName(name);
        x->setElemType(TensorProto_DataType_FLOAT);
        x->setSizes({Dimension(1024)});
        g->addInitializer(initializer);
      } else {
        x = g->addInitializerAndCreateValue(initializer);
      }
      Node* node1 = g->create(kNeg, 1);
      node1->addInput(x);
      g->appendNode(node1);
      Node* node2 = g->create(kNeg, 1);
      node2->addInput(node1->outputs()[0]);
      g->appendNode(node2);
      g->registerOutput(node2->outputs()[0]);

      ModelProto model;
      model.set_ir_version(ir_version);
      ExportModelProto(&model, g);
      for (const auto& node : model.graph().node()) {
        for (const auto& output_name : node.output()) {
          EXPECT_TRUE(IsValidIdentifier(output_name));
        }
      }
      EXPECT_TRUE(IsValidIdentifier(model.graph().initializer(0).name()));

      auto imported = ImportModelProto(model);
      ASSERT_TRUE(imported != nullptr);
      ASSERT_EQ(imported->initializers().size(), 1u);
      const auto& tensor = imported->initializers().front();
      EXPECT_EQ(tensor.name(), name);
      EXPECT_EQ(tensor.raw(), raw_data);
      EXPECT_EQ(tensor.sizes(), std::vector<int64_t>({1024}));
      EXPECT_EQ(tensor.elem_type(), TensorProto_DataType_FLOAT);
      EXPECT_EQ(imported->initializer_names(), std::vector<std::string>({name}));
      ASSERT_EQ(imported->outputs().size(), 1u);
      const auto* last_node = imported->outputs()[0]->node();
      ASSERT_EQ(last_node->inputs().size(), 1u);
      const auto* first_node = last_node->inputs()[0]->node();
      ASSERT_EQ(first_node->inputs().size(), 1u);
      EXPECT_EQ(first_node->inputs()[0]->uniqueName(), name);
      EXPECT_EQ(model.graph().initializer(0).raw_data(), raw_data);
      EXPECT_EQ(model.graph().initializer(0).name(), name);
    }
  }
}

// Regression: copyMetadata() must not turn "rank unknown" into rank 0.
TEST(IR, CopyMetadataPreservesUnknownRank) {
  Graph g;
  g.setName("test");
  Value* from = g.addInput();
  Value* to = g.addInput();
  to->setSizes({Dimension(1)});
  ASSERT_FALSE(from->has_sizes());
  ASSERT_TRUE(to->has_sizes());

  to->copyMetadata(from);
  EXPECT_FALSE(to->has_sizes());
}

// copyMetadata() must still copy a known rank/shape over.
TEST(IR, CopyMetadataCopiesKnownSizes) {
  Graph g;
  g.setName("test");
  Value* from = g.addInput();
  from->setSizes({Dimension(2), Dimension(3)});
  Value* to = g.addInput();
  ASSERT_FALSE(to->has_sizes());

  to->copyMetadata(from);
  ASSERT_TRUE(to->has_sizes());
  ASSERT_EQ(to->sizes().size(), 2u);
  EXPECT_EQ(to->sizes()[0].dim, 2);
  EXPECT_EQ(to->sizes()[1].dim, 3);
}

// Regression tests for Graph's name-uniqueness bookkeeping (used_names_ /
// subgraph_bearing_nodes_, backing isNameUnique()/getNextUniqueName()): a
// name can have more than one live holder at once (an initializer's Tensor
// entry and its mirrored graph Value, or an output value mid-rename in
// Value::replaceAllUsesWith), and releasing just one holder must not free
// the name while another is still displaying it.

// getNextUniqueName() only ever proposes "_v_<n>" candidates for a
// strictly-increasing, never-reused n, so a name it might mint again later
// has to itself be "_v_<n>"-shaped for some not-yet-reached n -- an
// arbitrary held name (e.g. "y") is never revisited regardless of whether
// it's correctly reserved. These tests reserve such a not-yet-reached slot
// explicitly, ahead of the counter's current position, then drive the
// counter up to it via ordinary getNextUniqueName() calls: with the name
// still correctly reserved, that exact "_v_<n>" is skipped over; with the
// bug, it gets minted a second time.

// Value::replaceAllUsesWith() on a registered graph output renames the old
// output value off of its name so the replacement can take it over. The old
// value's release of that name must not un-reserve it while the replacement
// is still live and displaying it.
TEST(IR, ReplaceAllUsesWithKeepsGraphOutputNameReserved) {
  Graph g;
  g.setName("test");
  Value* x = g.addInput();
  x->setUniqueName("x");

  Node* node1 = g.create(kNeg, 1);
  node1->addInput(x);
  g.appendNode(node1);
  Value* y = node1->outputs()[0];

  Node* node2 = g.create(kNeg, 1);
  node2->addInput(x);
  g.appendNode(node2);
  Value* replacement = node2->outputs()[0];

  const std::string y_name = toVarName(replacement->unique() + 5);
  y->setUniqueName(y_name);
  g.registerOutput(y);

  y->replaceAllUsesWith(replacement);
  ASSERT_EQ(replacement->uniqueName(), y_name);

  bool saw_reserved_name_again = false;
  for (int i = 0; i < 20; ++i) {
    if (g.getNextUniqueName() == y_name) {
      saw_reserved_name_again = true;
    }
  }
  EXPECT_FALSE(saw_reserved_name_again);
}

// isNameUnique() recurses into If/Loop/Scan subgraph bodies to avoid minting
// a name in the parent graph that a nested subgraph's own value already
// displays. Each Graph -- parent and subgraph alike -- keeps an
// independently-numbered id counter, so an unnamed ("_v_<n>") value inside a
// subgraph can share its default display name with a value the parent graph
// is about to mint, purely by counter coincidence.
TEST(IR, IsNameUniqueSeesDefaultNamesInsideSubgraphs) {
  Graph parent;
  parent.setName("parent");
  Value* cond = parent.addInput();
  cond->setUniqueName("cond");

  auto then_graph = std::make_shared<Graph>();
  then_graph->setName("then");
  Value* then_in = then_graph->addInput();
  then_in->setUniqueName("then_in");
  Node* then_node = then_graph->create(kNeg, 1);
  then_node->addInput(then_in);
  then_graph->appendNode(then_node);
  Value* then_unnamed = then_node->outputs()[0]; // never explicitly renamed
  then_graph->registerOutput(then_unnamed);
  const std::string collision_name = then_unnamed->uniqueName();

  Node* if_node = parent.create(kIf, 0);
  if_node->addInput(cond);
  parent.appendNode(if_node);
  if_node->g_(Symbol("then_branch"), then_graph);

  for (int i = 0; i < 20; ++i) {
    EXPECT_NE(parent.getNextUniqueName(), collision_name);
  }
}

// eraseInitializer() removes the Tensor-level bookkeeping for an
// initializer and, if it finds a matching output on initializer_node_,
// erases that too -- but for an IR<4 (or input-shadowed) initializer, the
// value holding that name is a *graph input* added separately via
// addInitializer() alone (mirroring ir_pb_converter.cc's import path: see
// its ir_version < 4 / "exists in input" branch), not a value under
// initializer_node_. eraseInitializer() must not touch that value's name.
TEST(IR, EraseInitializerKeepsNameReservedForLiveValue) {
  Graph g;
  g.setName("test");
  const std::string w_name = toVarName(5);

  Value* v = g.addInput();
  v->setUniqueName(w_name);
  Tensor t;
  t.setName(w_name);
  g.addInitializer(t);

  g.eraseInitializer(w_name);
  ASSERT_EQ(v->uniqueName(), w_name); // v itself must be untouched
  bool saw_reserved_name_again = false;
  for (int i = 0; i < 10; ++i) {
    if (g.getNextUniqueName() == w_name) {
      saw_reserved_name_again = true;
    }
  }
  EXPECT_FALSE(saw_reserved_name_again);
}

// clearInitializers() only clears the Tensor-level initializer bookkeeping;
// per its documented behavior, initializer_node_'s output Values survive and
// keep displaying their names, which must stay reserved.
TEST(IR, ClearInitializersKeepsSurvivingValueNamesReserved) {
  Graph g;
  g.setName("test");
  Tensor t;
  const std::string w_name = toVarName(5);
  t.setName(w_name);
  Value* v = g.addInitializerAndCreateValue(t);
  ASSERT_EQ(v->uniqueName(), w_name);

  g.clearInitializers();
  bool saw_reserved_name_again = false;
  for (int i = 0; i < 10; ++i) {
    if (g.getNextUniqueName() == w_name) {
      saw_reserved_name_again = true;
    }
  }
  EXPECT_FALSE(saw_reserved_name_again);
}

// forEachNode()'s subgraph walk must tolerate a callback that reaches back
// and mutates the attributes of a node it is currently recursing through --
// e.g. a rewrite pass editing its own enclosing If/Loop node while visiting
// a node inside that node's subgraph. This must not corrupt the walk (under
// ASAN: not a heap-use-after-free) and must still visit every node.
TEST(IR, ForEachNodeSurvivesSelfMutationDuringSubgraphWalk) {
  Graph parent;
  parent.setName("parent");
  Value* cond = parent.addInput();
  cond->setUniqueName("cond");

  auto body = std::make_shared<Graph>();
  body->setName("body");
  Value* body_in = body->addInput();
  body_in->setUniqueName("body_in");
  Node* inner = body->create(kNeg, 1);
  inner->addInput(body_in);
  body->appendNode(inner);
  body->registerOutput(inner->outputs()[0]);

  Node* if_node = parent.create(kIf, 0);
  if_node->addInput(cond);
  parent.appendNode(if_node);
  // A few unrelated attributes first, so the reentrant add below is more
  // likely to force a reallocation of if_node's own attribute storage.
  if_node->i_(Symbol("a1"), 1);
  if_node->i_(Symbol("a2"), 2);
  if_node->i_(Symbol("a3"), 3);
  if_node->g_(Symbol("then_branch"), body);

  int visited = 0;
  parent.forEachNode([&](Node* node) {
    ++visited;
    if (node == inner) {
      if_node->i_(Symbol("extra_attr"), 4);
    }
  });
  EXPECT_EQ(visited, 2);
  EXPECT_TRUE(if_node->hasAttribute(Symbol("extra_attr")));
}

// Regression test: Tensor::elem_num() and size_from_dim() must use 64-bit
// arithmetic. Previously, std::accumulate used `1` (int) as the initial value,
// causing 32-bit multiplication that silently overflowed for tensors whose
// element count exceeded INT_MAX (~2.1B). Fixed by using int64_t{1}.
TEST(Tensor, ElemNumLargeTensorNoOverflow) {
  Tensor t;
  // 50000 * 50000 = 2,500,000,000 which exceeds INT32_MAX (2,147,483,647)
  t.sizes() = {50000, 50000};
  const int64_t expected = static_cast<int64_t>(50000) * 50000;
  EXPECT_EQ(t.elem_num(), expected);
  EXPECT_EQ(t.size_from_dim(0), expected);
  EXPECT_EQ(t.size_from_dim(1), int64_t{50000});
}

// Build a raw_data string from native bytes of the given values.
template <typename T>
static std::string MakeRawData(const std::vector<T>& values) {
  std::string raw;
  raw.resize(values.size() * sizeof(T));
  std::memcpy(raw.data(), values.data(), raw.size());
  return raw;
}

// Regression: raw size not a multiple of the element size used to overflow.
#ifndef ONNX_NO_EXCEPTIONS
TEST(Tensor, ParseDataRawSizeNotMultipleThrows) {
  Tensor t;
  // 5 bytes is not a multiple of sizeof(int32_t) == 4.
  t.set_raw_data(std::string(5, '\0'));
  EXPECT_THROW(ParseData<int32_t>(&t), assert_error);
}
#endif

// Valid raw tensor round-trips; byte-symmetric values are endian-independent.
TEST(Tensor, ParseDataRawValid) {
  const std::vector<int32_t> values = {0, 0x01010101, 0x7F7F7F7F};
  Tensor t;
  t.set_raw_data(MakeRawData(values));
  EXPECT_EQ(ParseData<int32_t>(&t), values);
}

// Empty raw_data is a multiple of any element size and yields no elements.
TEST(Tensor, ParseDataRawEmpty) {
  Tensor t;
  t.set_raw_data(std::string());
  EXPECT_TRUE(ParseData<int32_t>(&t).empty());
}

} // namespace ONNX_NAMESPACE::Test
