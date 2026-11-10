// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/traditionalml/utils.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

ONNX_ML_OPERATOR_SET_SCHEMA(ArrayFeatureExtractor, 1, generated::ai_onnx_ml::ArrayFeatureExtractor_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(Binarizer, 1, generated::ai_onnx_ml::Binarizer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(CastMap, 1, generated::ai_onnx_ml::CastMap_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(CategoryMapper, 1, generated::ai_onnx_ml::CategoryMapper_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(DictVectorizer, 1, generated::ai_onnx_ml::DictVectorizer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(FeatureVectorizer, 1, generated::ai_onnx_ml::FeatureVectorizer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(Imputer, 1, generated::ai_onnx_ml::Imputer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(LabelEncoder, 4, generated::ai_onnx_ml::LabelEncoder_ver4());

ONNX_ML_OPERATOR_SET_SCHEMA(LinearClassifier, 1, generated::ai_onnx_ml::LinearClassifier_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(LinearRegressor, 1, generated::ai_onnx_ml::LinearRegressor_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(Normalizer, 1, generated::ai_onnx_ml::Normalizer_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(OneHotEncoder, 1, generated::ai_onnx_ml::OneHotEncoder_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(Scaler, 1, generated::ai_onnx_ml::Scaler_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(SVMClassifier, 1, generated::ai_onnx_ml::SVMClassifier_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(SVMRegressor, 1, generated::ai_onnx_ml::SVMRegressor_ver1());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleClassifier, 5, generated::ai_onnx_ml::TreeEnsembleClassifier_ver5());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleRegressor, 5, generated::ai_onnx_ml::TreeEnsembleRegressor_ver5());

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsemble, 5, generated::ai_onnx_ml::TreeEnsemble_ver5());

ONNX_ML_OPERATOR_SET_SCHEMA(ZipMap, 1, generated::ai_onnx_ml::ZipMap_ver1());

} // namespace ONNX_NAMESPACE
#endif
