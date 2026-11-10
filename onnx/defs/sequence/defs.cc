// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/sequence/utils.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    SequenceEmpty,
    11,
    generated::ai_onnx::SequenceEmpty_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::SequenceEmptyShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SequenceConstruct,
    11,
    generated::ai_onnx::SequenceConstruct_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::SequenceConstructShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SequenceInsert,
    11,
    generated::ai_onnx::SequenceInsert_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::SequenceInsertShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SequenceAt,
    11,
    generated::ai_onnx::SequenceAt_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::SequenceAtShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SequenceErase,
    11,
    generated::ai_onnx::SequenceErase_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::SequenceEraseShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SequenceLength,
    11,
    generated::ai_onnx::SequenceLength_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::SequenceLengthShapeInference_opset11));

// Updated operators that consume/produce sequence of tensors.

ONNX_OPERATOR_SET_SCHEMA(
    SplitToSequence,
    24,
    generated::ai_onnx::SplitToSequence_ver24().FillUsing(defs::sequence::utils::SplitToSequenceOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    ConcatFromSequence,
    11,
    generated::ai_onnx::ConcatFromSequence_ver11().TypeAndShapeInferenceFunction(
        defs::sequence::utils::ConcatFromSequenceShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    SequenceMap,
    17,
    generated::ai_onnx::SequenceMap_ver17()
        .SetContextDependentFunctionBodyBuilder(defs::sequence::utils::BuildSequenceMapBodyFunc)
        .TypeAndShapeInferenceFunction(defs::sequence::utils::SequenceMapInferenceFunction));

} // namespace ONNX_NAMESPACE
