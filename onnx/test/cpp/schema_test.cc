/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "gtest/gtest.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace Test {

TEST(SchemaTest, RegisterCertainOpsetSchema) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
    EXPECT_TRUE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);
    RegisterOnnxOperatorSetSchema(13);
    EXPECT_TRUE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 13);

    auto opSchema = OpSchemaRegistry::Schema("Add");
    EXPECT_NE(nullptr, opSchema);
    EXPECT_EQ(opSchema->SinceVersion, 13);

#endif
}
} // namespace Test
} // namespace ONNX_NAMESPACE
