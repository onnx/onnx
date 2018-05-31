// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
using namespace ONNX_NAMESPACE;


ONNX_FUNCTION(FunctionBuilder()
  .SetDomain("")
  .SetBuildFunction(
    BuildFunction(new std::unique_ptr<FunctionProto>())
  )
);