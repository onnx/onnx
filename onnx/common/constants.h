// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <string>

namespace ONNX_NAMESPACE {
// For ONNX op/function registration.

// ONNX domains.
constexpr const char* AI_ONNX_ML_DOMAIN = "ai.onnx.ml";
constexpr const char* ONNX_DOMAIN = "";
constexpr bool OPTIONAL = false;

// For dimension denotation.
constexpr const char* DATA_BATCH = "DATA_BATCH";
constexpr const char* DATA_CHANNEL = "DATA_CHANNEL";
constexpr const char* DATA_TIME = "DATA_TIME";
constexpr const char* DATA_FEATURE = "DATA_FEATURE";
constexpr const char* FILTER_IN_CHANNEL = "FILTER_IN_CHANNEL";
constexpr const char* FILTER_OUT_CHANNEL = "FILTER_OUT_CHANNEL";
constexpr const char* FILTER_SPATIAL = "FILTER_SPATIAL";

// For type denotation.
constexpr const char* TENSOR = "TENSOR";
constexpr const char* IMAGE = "IMAGE";
constexpr const char* AUDIO = "AUDIO";
constexpr const char* TEXT = "TEXT";

} // namespace ONNX_NAMESPACE
