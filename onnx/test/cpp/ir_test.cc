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
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace Test {

static bool IsValidIdentifier(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  if (!isalpha(name[0]) && name[0] != '_') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isalnum(name[i]) && name[i] != '_') {
      return false;
    }
  }
  return true;
}

TEST(IR, ValidIdentifierTest) {
  Graph* g = new Graph(); // NOLINT(cppcoreguidelines-owning-memory)
  g->setName("test");
  Value* x = g->addInput();
  x->setUniqueName("x");
  x->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  x->setSizes({Dimension("M"), Dimension("N")});
  Node* node1 = g->create(kNeg, 1);
  node1->addInput(x);
  g->appendNode(node1);
  Value* temp1 = node1->outputs()[0];
  Node* node2 = g->create(kNeg, 1);
  node2->addInput(temp1);
  g->appendNode(node2);
  Value* y = node2->outputs()[0];
  g->registerOutput(y);

  ModelProto model;
  ExportModelProto(&model, std::shared_ptr<Graph>(g));

  for (const auto& node : model.graph().node()) {
    for (const auto& name : node.output()) {
      EXPECT_TRUE(IsValidIdentifier(name));
    }
  }
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

} // namespace Test
} // namespace ONNX_NAMESPACE
