// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/controlflow/utils.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    If,
    24,
    generated::ai_onnx::If_ver24().TypeAndShapeInferenceFunction(defs::controlflow::utils::IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    23,
    generated::ai_onnx::If_ver23().TypeAndShapeInferenceFunction(defs::controlflow::utils::IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    21,
    generated::ai_onnx::If_ver21().TypeAndShapeInferenceFunction(defs::controlflow::utils::IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    19,
    generated::ai_onnx::If_ver19().TypeAndShapeInferenceFunction(defs::controlflow::utils::IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    16,
    generated::ai_onnx::If_ver16().TypeAndShapeInferenceFunction(defs::controlflow::utils::IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    24,
    generated::ai_onnx::Loop_ver24().TypeAndShapeInferenceFunction(defs::controlflow::utils::LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    23,
    generated::ai_onnx::Loop_ver23().TypeAndShapeInferenceFunction(defs::controlflow::utils::LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    21,
    generated::ai_onnx::Loop_ver21().TypeAndShapeInferenceFunction(defs::controlflow::utils::LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    19,
    generated::ai_onnx::Loop_ver19().TypeAndShapeInferenceFunction(defs::controlflow::utils::LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    16,
    generated::ai_onnx::Loop_ver16().TypeAndShapeInferenceFunction(defs::controlflow::utils::LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    24,
    generated::ai_onnx::Scan_ver24().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    23,
    generated::ai_onnx::Scan_ver23().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    21,
    generated::ai_onnx::Scan_ver21().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    19,
    generated::ai_onnx::Scan_ver19().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    16,
    generated::ai_onnx::Scan_ver16().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction)); // Shares same shape inference as opset 11

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    8,
    generated::ai_onnx::Scan_ver8().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction_opset8));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    1,
    generated::ai_onnx::Loop_ver1().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::LoopInferenceFunction_opset8));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    11,
    generated::ai_onnx::Loop_ver11().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::LoopInferenceFunction_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    9,
    generated::ai_onnx::Scan_ver9().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::ScanInferenceFunction_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    1,
    generated::ai_onnx::If_ver1().TypeAndShapeInferenceFunction(defs::controlflow::utils::IfInferenceFunction_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    11,
    generated::ai_onnx::If_ver11().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::IfInferenceFunction_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    If,
    13,
    generated::ai_onnx::If_ver13().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::IfInferenceFunction_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    13,
    generated::ai_onnx::Loop_ver13().TypeAndShapeInferenceFunction(
        defs::controlflow::utils::LoopInferenceFunction_opset13));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    11,
    generated::ai_onnx::Scan_ver11().TypeAndShapeInferenceFunction(defs::controlflow::utils::ScanInferenceFunction));

} // namespace ONNX_NAMESPACE
