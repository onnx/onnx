// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace inliner {

// IR version 10 introduces overloaded function names. The following APIs to specify
// functions to be inlined currently allow only specifying (domain, name). Thus,
// either all overloads of a function are inlined or none.
// The older-style ids are used below for backward compatibility.

// A FunctionId is a pair of strings (domain, function name).
using FunctionId = std::pair<std::string, std::string>;

// A vector of FunctionIds.
using FunctionIdVector = std::vector<FunctionId>;

// Interface used to represent a set of function ids for the inliner.
class FunctionIdSet {
 public:
  virtual bool Contains(const std::string& function_domain, const std::string& function_name) const = 0;
  virtual ~FunctionIdSet() = default;

  // Factory methods for creating FunctionIdSet instances.

  // Creates a set representing the elements in the given vector, if invert is false.
  // Otherwise, creates a set representing elements not in the given vector.
  static std::unique_ptr<FunctionIdSet> Create(FunctionIdVector&& function_ids, bool invert = false);
};

/**
 * @brief Inlines the schema-defined functions in the given model that are in the given set.
 * @param model The model in which functions will be inlined.
 * @param to_inline The set of functions to inline.
 * @param schema_registry The schema registry used for function lookup. If nullptr,
 *        the default schema registry is used.
 * @note Only call-sites in the main graph are inlined.
 */
void InlineSelectedFunctions(ModelProto& model, const FunctionIdSet& to_inline, const ISchemaRegistry* schema_registry);

/**
 * @brief Inlines the model-local functions in the given model that are in the given set.
 *
 * This function processes the model and replaces all call-sites of the specified
 * model-local functions with their inlined implementations. The inlined functions
 * are also removed from the model's list of functions.
 *
 * @param model The model in which functions will be inlined.
 * @param to_inline The set of functions to inline.
 *
 * @note This function does not perform schema-defined function inlining. For schema-defined
 *       function inlining, use InlineSelectedFunctions instead.
 */
void InlineSelectedLocalFunctions(ModelProto& model, const FunctionIdSet& to_inline);

/**
 * @brief Inlines the model-local functions in the given model that are in the given set.
 * @deprecated This function is deprecated. Use InlineSelectedLocalFunctions instead,
 * to avoid confusion with the overloaded version of InlineSelectedFunctions that
 * inlines schema-defined functions as well.
 */
void InlineSelectedFunctions(ModelProto& model, const FunctionIdSet& to_inline);

// Inlines all model-local functions in the given model. This supports version
// conversion, an advanced feature that is not enabled by default. When enabled,
// the inliner will attempt to convert the version of the inlined function to
// match the version of the model. If not enabled, the inliner will only inline
// functions that use opset versions that are compatible with the model.
void InlineLocalFunctions(ModelProto& model, bool convert_version = false);

} // namespace inliner
} // namespace ONNX_NAMESPACE
