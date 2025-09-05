// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "catch2/catch_test_macros.hpp"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace Test {

TEST_CASE("SchemaRegistrationTest", "[DisabledOnnxStaticRegistrationAPICall]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  REQUIRE(IsOnnxStaticRegistrationDisabled());
#else
  REQUIRE_FALSE(IsOnnxStaticRegistrationDisabled());
#endif
}

// Schema of all versions are registered by default
// Further schema manipulation expects to be error-free
TEST_CASE("SchemaRegistrationTest", "[RegisterAllByDefaultAndManipulateSchema]") {
#ifndef __ONNX_DISABLE_STATIC_REGISTRATION

  // Expects all opset registered by default
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 0);

  // Should find schema for all versions
  REQUIRE(OpSchemaRegistry::Schema("Add", 1));
  REQUIRE(OpSchemaRegistry::Schema("Add", 6));
  REQUIRE(OpSchemaRegistry::Schema("Add", 7));
  REQUIRE(OpSchemaRegistry::Schema("Add", 13));

  // Clear all opset schema registration
  DeregisterOnnxOperatorSetSchema();

  // Should not find any opset
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Add"));

  // Register all opset versions
  RegisterOnnxOperatorSetSchema();

  // Should find all opset
  REQUIRE(OpSchemaRegistry::Schema("Add"));
#endif
}

// By default ONNX registers all opset versions and selective schema loading cannot be tested
// So these tests are run only when static registration is disabled
TEST_CASE("SchemaRegistrationTest", "[RegisterAndDeregisterAllOpsetSchemaVersion]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION

  // Clear all opset schema registration
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);

  // Should not find schema for any op
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Acos"));
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Add"));
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Trilu"));

  // Register all opset versions
  RegisterOnnxOperatorSetSchema(0);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 0);

  // Should find schema for all ops. Available versions are:
  // Acos-7
  // Add-1,6,7,13,14
  // Trilu-14
  auto schema = OpSchemaRegistry::Schema("Acos", 7);
  REQUIRE(schema);
  REQUIRE(schema->SinceVersion() == 7);

  schema = OpSchemaRegistry::Schema("Add", 14);
  REQUIRE(schema);
  REQUIRE(schema->SinceVersion() == 14);

  schema = OpSchemaRegistry::Schema("Trilu");
  REQUIRE(schema);
  REQUIRE(schema->SinceVersion() == 14);

  REQUIRE(OpSchemaRegistry::Schema("Add", 1));
  REQUIRE(OpSchemaRegistry::Schema("Add", 6));
  REQUIRE(OpSchemaRegistry::Schema("Add", 7));
  REQUIRE(OpSchemaRegistry::Schema("Add", 13));

  // Clear all opset schema registration
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);

  // Should not find schema for any op
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Acos"));
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Add"));
  REQUIRE_FALSE(OpSchemaRegistry::Schema("Trilu"));
#endif
}

TEST_CASE("SchemaRegistrationTest", "[RegisterSpecifiedOpsetSchemaVersion]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);
  RegisterOnnxOperatorSetSchema(13);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 13);

  auto opSchema = OpSchemaRegistry::Schema("Add");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 13);

  // Should not find opset 12
  opSchema = OpSchemaRegistry::Schema("Add", 12);
  REQUIRE_FALSE(opSchema);

  // Should not find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  REQUIRE_FALSE(opSchema);

  // Acos-7 is the latest Acos before specified 13
  opSchema = OpSchemaRegistry::Schema("Acos", 13);
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 7);
#endif
}

// Register opset-11, then opset-14
// Expects Reg(11, 14) == Reg(11) U Reg(14)
TEST_CASE("SchemaRegistrationTest", "[RegisterMultipleOpsetSchemaVersions_UpgradeVersion]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);

  // Register opset 11
  RegisterOnnxOperatorSetSchema(11);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 11);
  // Register opset 14
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(14, false);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 14);

  // Acos-7 is the latest before/at opset 11 and 14
  auto opSchema = OpSchemaRegistry::Schema("Acos");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 7);

  // Add-7 is the latest before/at opset 11
  // Add-14 is the latest before/at opset 14
  // Should find both Add-7,14
  REQUIRE(OpSchemaRegistry::Schema("Add", 7));
  REQUIRE(OpSchemaRegistry::Schema("Add", 14));

  // Should find the max version 14
  opSchema = OpSchemaRegistry::Schema("Add");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 14);

  // Should find Add-7 as the max version <=13
  opSchema = OpSchemaRegistry::Schema("Add", 13);
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 7);

  // Should find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 14);
#endif
}

// Register opset-14, then opset-11
// Expects Reg(14, 11) == Reg(11) U Reg(14)
TEST_CASE("SchemaRegistrationTest", "[RegisterMultipleOpsetSchemaVersions_DowngradeVersion]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);

  // Register opset 14
  RegisterOnnxOperatorSetSchema(14);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 14);
  // Register opset 11
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(11, false);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 11);

  // Acos-7 is the latest before/at opset 11 and 14
  auto opSchema = OpSchemaRegistry::Schema("Acos");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 7);

  // Add-7 is the latest before/at opset 11
  // Add-14 is the latest before/at opset 14
  // Should find both Add-7,14
  REQUIRE(OpSchemaRegistry::Schema("Add", 7));
  REQUIRE(OpSchemaRegistry::Schema("Add", 14));

  // Should find the max version 14
  opSchema = OpSchemaRegistry::Schema("Add");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 14);

  // Should find Add-7 as the max version <=13
  opSchema = OpSchemaRegistry::Schema("Add", 13);
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 7);

  // Should find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  REQUIRE(opSchema);
  REQUIRE(opSchema->SinceVersion() == 14);
#endif
}

// Register opset-11, then all versions
// Expects no error
TEST_CASE("SchemaRegistrationTest", "[RegisterSpecificThenAllVersion]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);

  // Register opset 11
  RegisterOnnxOperatorSetSchema(11);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 11);

  // Register all opset versions
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(0, false);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 0);

  // Should find schema for all ops
  REQUIRE(OpSchemaRegistry::Schema("Acos"));
  REQUIRE(OpSchemaRegistry::Schema("Add"));
  REQUIRE(OpSchemaRegistry::Schema("Trilu"));

  // Should find schema for all versions
  REQUIRE(OpSchemaRegistry::Schema("Add", 1));
  REQUIRE(OpSchemaRegistry::Schema("Add", 6));
  REQUIRE(OpSchemaRegistry::Schema("Add", 7));
  REQUIRE(OpSchemaRegistry::Schema("Add", 13));
#endif
}

// Register all versions, then opset 11
// Expects no error
TEST_CASE("SchemaRegistrationTest", "[RegisterAllThenSpecificVersion]") {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);

  // Register all opset versions
  RegisterOnnxOperatorSetSchema(0);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 0);

  // Register opset 11
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(11, false);
  REQUIRE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 11);

  // Should find schema for all ops
  REQUIRE(OpSchemaRegistry::Schema("Acos"));
  REQUIRE(OpSchemaRegistry::Schema("Add"));
  REQUIRE(OpSchemaRegistry::Schema("Trilu"));

  // Should find schema for all versions
  REQUIRE(OpSchemaRegistry::Schema("Add", 1));
  REQUIRE(OpSchemaRegistry::Schema("Add", 6));
  REQUIRE(OpSchemaRegistry::Schema("Add", 7));
  REQUIRE(OpSchemaRegistry::Schema("Add", 13));
#endif
}

} // namespace Test
} // namespace ONNX_NAMESPACE
