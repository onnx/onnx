// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#define ONNX_UNUSED_PARAMETER(x) (void)(x)

#define ONNX_RETURN_IF_ERROR(expr) \
  do {                             \
    auto _status = (expr);         \
    if ((!_status.IsOK()))         \
      return _status;              \
  } while (0)
