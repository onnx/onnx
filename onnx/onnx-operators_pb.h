// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "onnx/onnx_pb.h"
#ifdef ONNX_ML
#include "onnx/onnx-operators-ml.pb.h"
#else
#include "onnx/onnx-operators.pb.h"
#endif