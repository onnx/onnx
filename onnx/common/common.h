// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

namespace ONNX_NAMESPACE {
#ifdef _WIN32
#define ONNX_UNUSED_PARAMETER(x) (x)
#else
#define ONNX_UNUSED_PARAMETER(x) (void)(x)
#endif
} // namespace ONNX_NAMESPACE
