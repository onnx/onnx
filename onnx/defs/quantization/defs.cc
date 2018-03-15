// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

ONNX_OPERATOR_SCHEMA(Quantize)
    .SetDoc(R"DOC(Quantize a float tensor into a uint8 quantized tensor)DOC")
    .Input(0, "input", "Input tensor of any shape.", "T0")
    .Output(0, "output", "Output tensor of same shape and type as input.", "T1")
    .TypeConstraint(
        "T0",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input types to float tensors.")
    .TypeConstraint(
        "T1",
        {"tensor(uint8)"},
        "Constrain output types to uint8 tensors."); 

ONNX_OPERATOR_SCHEMA(Dequantize)
    .SetDoc(R"DOC(Dequantize a uint8 quantized tensor into a float tensor)DOC")
    .Input(0, "input", "Input tensor of any shape.", "T0")
    .Output(0, "output", "Output tensor of same shape and type as input.", "T1")
    .TypeConstraint(
        "T0",
        {"tensor(uint8)"},
        "Constrain input types to uint8 tensors.")
    .TypeConstraint(
        "T1",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain output types to float tensors.");                
