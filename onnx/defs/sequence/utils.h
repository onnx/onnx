// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE::defs::sequence::utils {

void SplitToSequenceShapeInference(InferenceContext& ctx);
void SequenceEmptyShapeInference_opset11(InferenceContext& ctx);
void SequenceConstructShapeInference_opset11(InferenceContext& ctx);
void SequenceInsertShapeInference_opset11(InferenceContext& ctx);
void SequenceAtShapeInference_opset11(InferenceContext& ctx);
void SequenceEraseShapeInference_opset11(InferenceContext& ctx);
void SequenceLengthShapeInference_opset11(InferenceContext& ctx);
void ConcatFromSequenceShapeInference_opset11(InferenceContext& ctx);
void SequenceMapInferenceFunction(InferenceContext& ctx);
bool BuildSequenceMapBodyFunc(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);

} // namespace ONNX_NAMESPACE::defs::sequence::utils
