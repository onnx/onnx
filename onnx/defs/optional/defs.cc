// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/optional/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static std::vector<std::string> tensor_and_sequence_types_ir14() {
  return types::Concat(OpSchema::all_tensor_types_ir14(), OpSchema::all_tensor_sequence_types_ir14());
}

ONNX_OPERATOR_SET_SCHEMA(
    Optional,
    28,
    OpSchema().FillUsing(
        defs::optional::utils::OptionalOpGenerator(
            tensor_and_sequence_types_ir14(),
            types::Optional(tensor_and_sequence_types_ir14()))));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    28,
    OpSchema().FillUsing(
        defs::optional::utils::OptionalHasElementOpGenerator(
            types::Concat(types::Optional(tensor_and_sequence_types_ir14()), tensor_and_sequence_types_ir14()))));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    28,
    OpSchema().FillUsing(
        defs::optional::utils::OptionalGetElementOpGenerator(
            types::Optional(tensor_and_sequence_types_ir14()),
            tensor_and_sequence_types_ir14())));
} // namespace ONNX_NAMESPACE
