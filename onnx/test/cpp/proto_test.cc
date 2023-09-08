// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"
#include "onnx/onnx_pb.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace Test {

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

TEST(ProtoTest, Move) {
    NodeList nodelist1;
    NodeList nodelist2;

    nodelist1.Add()->set_name("node1");
    nodelist1.Add()->set_name("node2");

    ASSERT_EQ(nodelist1.size(), 2);
    ASSERT_EQ(nodelist2.size(), 0);

    nodelist2 = std::move(nodelist1);

    ASSERT_EQ(nodelist1.size(), 0);
    ASSERT_EQ(nodelist2.size(), 2);

    NodeProto node1;
    NodeProto node2;

    node1.mutable_input()->Add("input1");
    node1.mutable_input()->Add("input2");

    ASSERT_EQ(node1.input().size(), 2);
    ASSERT_EQ(node2.input().size(), 0);

    node2 = std::move(node1);

    ASSERT_EQ(node1.input().size(), 0);
    ASSERT_EQ(node2.input().size(), 2);

    nodelist1.Add(std::move(node2));
    ASSERT_EQ(nodelist1.size(), 1);
    ASSERT_EQ(nodelist1.Get(0).input().size(), 2);
    ASSERT_EQ(node2.input().size(), 0);
}

}
}
