// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_DEFS_TRADITIONALML_UTILS_H_
#define ONNX_DEFS_TRADITIONALML_UTILS_H_

#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

inline void AssertAttributeProtoTypeAndLength(
    const AttributeProto* attr_proto,
    int expected_length,
    TensorProto_DataType expected_type,
    bool required) {
  if (nullptr == attr_proto) {
    if (required) {
      fail_shape_inference("Unspecified required attribute.");
    }
    return;
  }
  const auto& [type, length] = getAttributeProtoElemTypeAndLength(attr_proto);
  if (type != expected_type) {
    fail_shape_inference(
        "Attribute '", attr_proto->name(), "' must have type ", TensorProto_DataType_Name(expected_type), ".");
  }
  if (length != expected_length) {
    fail_shape_inference("Attribute '", attr_proto->name(), "' must have ", expected_length, " elements.");
  }
}

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE::defs::traditionalml::utils {

void ArrayFeatureExtractorShapeInference_opset1(InferenceContext& ctx);
void CastMapShapeInference_opset1(InferenceContext& ctx);
void CategoryMapperShapeInference_opset1(InferenceContext& ctx);
void DictVectorizerShapeInference_opset1(InferenceContext& ctx);
void LabelEncoderShapeInference_opset4(InferenceContext& ctx);
void LinearClassifierShapeInference_opset1(InferenceContext& ctx);
void OneHotEncoderShapeInference_opset1(InferenceContext& ctx);
void SVMClassifierShapeInference_opset1(InferenceContext& ctx);
void TreeEnsembleShapeInference_opset5(InferenceContext& ctx);
void ZipMapShapeInference_opset1(InferenceContext& ctx);
void LabelEncoderShapeInference_opset1(InferenceContext& ctx);
void TreeEnsembleClassifierShapeInference_opset1(InferenceContext& ctx);
void TreeEnsembleClassifierShapeInference_opset3(InferenceContext& ctx);
void TreeEnsembleRegressorShapeInference_opset3(InferenceContext& ctx);
void LabelEncoderShapeInference_opset2(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::traditionalml::utils

#endif // ONNX_DEFS_TRADITIONALML_UTILS_H_
