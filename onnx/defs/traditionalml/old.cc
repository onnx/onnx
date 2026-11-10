// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/traditionalml/utils.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

ONNX_ML_OPERATOR_SET_SCHEMA(LabelEncoder, 1, generated::ai_onnx_ml::LabelEncoder_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleClassifier, 1, generated::ai_onnx_ml::TreeEnsembleClassifier_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleClassifier, 3, generated::ai_onnx_ml::TreeEnsembleClassifier_ver3());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleRegressor, 1, generated::ai_onnx_ml::TreeEnsembleRegressor_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleRegressor, 3, generated::ai_onnx_ml::TreeEnsembleRegressor_ver3());

ONNX_ML_OPERATOR_SET_SCHEMA(LabelEncoder, 2, generated::ai_onnx_ml::LabelEncoder_ver2());
} // namespace ONNX_NAMESPACE
#endif
