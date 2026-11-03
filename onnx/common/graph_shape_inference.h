// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

// Runs ONNX type-and-shape inference directly against the C++ IR (Graph `g`),
// with no ModelProto <-> Graph round trip: each op's existing, schema-
// registered TypeAndShapeInferenceFunction runs unmodified via
// shape_inference::InferenceContextImpl, fed a lightweight per-node view, and
// results are merged back with onnx's own mergeShapesAndTypes. Lets a caller
// that already holds a resident Graph (e.g. an onnx-optimizer pass pipeline)
// run shape inference without converting to/from ModelProto.
//
// If/Loop/Scan bodies are inferred too: each GRAPH/GRAPHS attribute is
// exported to a GraphProto per node visit and wired through
// GraphInferenceContext, so the op's own inference function recurses into it
// via onnx's existing GraphInferencerImpl/InferShapesImpl.
//
// v1 scope -- unimplemented cases leave the node's outputs unchanged (safe,
// same as an unknown op), just less complete:
//  - Function-body inference (schema->HasFunction()) is not implemented.
//  - ShapeInferenceOptions::enable_data_propagation is not honored.
//  - Sparse tensor inputs are not fed to getInputSparseData().
//  - Symbolic shapes aren't unified across a Loop's iterations.
//
// Returns whether any value's inferred type/shape changed.
bool InferShapesOnGraph(Graph& g, const ShapeInferenceOptions& options = ShapeInferenceOptions());

} // namespace ONNX_NAMESPACE
