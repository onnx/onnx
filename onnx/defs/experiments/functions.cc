// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
using namespace ONNX_NAMESPACE;

std::unique_ptr<FunctionProto> *func_ptr = new std::unique_ptr<FunctionProto>();
func_ptr->reset(&function);
ONNX_FUNCTION(FunctionBuilder().SetDomain(domain).SetBuildFunction(BuildFunction(func_ptr)));