/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "gtest/gtest.h"
#include "onnx/defs/operator_sets.h"

namespace ONNX_NAMESPACE {
namespace Test {

TEST(SchemaTest, RegisterAllOpsetSchema) {
#ifndef ONNX_DISABLE_STATIC_REGISTRATION
    EXPECT_TRUE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);
    RegisterOnnxOperatorSetSchema();
    EXPECT_TRUE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 0);
    //EXPECT_TRUE(OpSchemaRegistry::GetRegisteredSchemaCount() == ONNX_DBG_GET_COUNT_IN_OPSETS());
#endif
}
} // namespace Test
} // namespace ONNX_NAMESPACE
