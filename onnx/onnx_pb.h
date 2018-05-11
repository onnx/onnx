// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ONNX_ONNX_PB_H
#define ONNX_ONNX_PB_H

/**
 * Macro for marking functions as having public visibility.
 * Ported from folly/CPortability.h
 */
#ifndef __GNUC_PREREQ
#if defined __GNUC__ && defined __GNUC_MINOR__
#define __GNUC_PREREQ(maj, min) \
  ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
#define __GNUC_PREREQ(maj, min) 0
#endif
#endif

// Defines ONNX_EXPORT and ONNX_IMPORT. On Windows, this corresponds to
// different declarations (dllexport and dllimport). On Linux/Mac, it just
// resolves to the same "default visibility" setting.
#if defined(_MSC_VER)
#if defined(ONNX_BUILD_SHARED_LIBS)
#define ONNX_EXPORT __declspec(dllexport)
#define ONNX_IMPORT __declspec(dllimport)
#else
#define ONNX_EXPORT
#define ONNX_IMPORT
#endif
#else
#if defined(__GNUC__)
#if __GNUC_PREREQ(4, 9)
#define ONNX_EXPORT [[gnu::visibility("default")]]
#else
#define ONNX_EXPORT __attribute__((__visibility__("default")))
#endif
#else
#define ONNX_EXPORT
#endif
#define ONNX_IMPORT ONNX_EXPORT
#endif

// ONNX_API is a macro that, depends on whether you are building the
// main ONNX library or not, resolves to either ONNX_EXPORT or
// ONNX_IMPORT.
//
// This is used in e.g. ONNX's protobuf files: when building the main library,
// it is defined as ONNX_EXPORT to fix a Windows global-variable-in-dll
// issue, and for anyone dependent on ONNX it will be defined as
// ONNX_IMPORT.
// This is a solution from https://github.com/caffe2/caffe2/blob/4f534fad1af9f77d4f0496ecd37dafb382330223/caffe2/core/common.h
#ifndef ONNX_API
#ifdef ONNX_BUILD_MAIN_LIB
#define ONNX_API ONNX_EXPORT
#else
#define ONNX_API ONNX_IMPORT
#endif
#endif

#ifdef ONNX_ML
#include "onnx/onnx-ml.pb.h"
#else
#include "onnx/onnx.pb.h"
#endif

#endif // ! ONNX_ONNX_PB_H
