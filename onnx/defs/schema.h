// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <climits>
#include <limits>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

#include "onnx/onnx.pb.h"

namespace onnx {

// A const value returned by OpSchema::CalculateOutput() if the number of
// output cannot be determined.
constexpr int kCannotComputeNumOutputs = -1;

/**
 * @brief A class to record the schema of an op.
 *
 * OpSchema records the common interface of an op specified by its name.
 *
 * To register an OpSchema, one can use the macro OPERATOR_SCHEMA(name) and
 * then append the various functions in the class. For example, for an op
 * that itakes in two inputs, one output, and the first input and output
 * could be in-place, can be written as
 *
 *     OPERATOR_SCHEMA(name)
 *         .NumInputs(2).NumOutputs(1).AllowConsumed({{0, 0}});
 */
class OpSchema {
 public:
  enum class SupportType {
    COMMON, // Supported by all frameworks that support this IR.
    EXPERIMENTAL, // This OP is experimental and can be changed or removed in the future.
  };

  OpSchema() : name_("unknown"), file_("unknown"), line_(0), support_(SupportType::COMMON) {}
  OpSchema(const std::string& name, const std::string& file, const int line)
      : name_(name), file_(file), line_(line), support_(SupportType::COMMON) {}

  /**
   * @brief Returns the file that the op schema is registered from.
   */
  inline const std::string& file() const { return file_; }

  /**
   * @brief Returns the line in file that the op schema is registered from.
   */
  inline int line() const { return line_; }

  /**
   * @brief Returns the support level of the op schema.
   */
  SupportType support_level() const { return support_; }

  /**
   * @brief Returns the docstring of the op schema.
   */
  inline const char* doc() const {
    return doc_.empty() ? nullptr : doc_.c_str();
  }

  /**
   * @brief Verifies if a NodeProto matches the pattern specified in
   * the schema.
   */
  bool Verify(const NodeProto& node) const;

  // Functions to set the property of the operator schemas.
  // Sets the number of inputs, either a fixed number or a min and a max.

  /**
   * @brief A single input.
   */
  OpSchema& NumInputs(int n);
  /**
   * @brief Input could be in range [min, max], inclusive.
   */
  OpSchema& NumInputs(int min, int max);
  /**
   * @brief Input could be one of the values specified in allowed_input_nums.
   */
  OpSchema& NumInputs(std::set<int> allowed_input_nums);
  /**
   * @brief Input is checked with a specified function.
   */
  OpSchema& NumInputs(std::function<bool(int)> func);

  // Sets the number of outputs, either a fixed number, a min and a max,
  // or a function that takes in the input number and produces an output
  // number. Use only one function in the set below.
  /**
   * @brief A single output.
   */
  OpSchema& NumOutputs(int n);
  /**
   * @brief Output could be in range [min, max], inclusive.
   */
  OpSchema& NumOutputs(int min, int max);
  /**
   * @brief Output could be one of the values specified in allowed_output_nums.
   */
  OpSchema& NumOutputs(std::set<int> allowed_output_nums);
  /**
   * @brief Output is checked with a specified function.
   */
  OpSchema& NumOutputs(std::function<bool(int)> func);

  /**
   * @brief Relationship between inputs and outputs is checked with a specified
   * function.
   */
  OpSchema& NumInputsOutputs(std::function<bool(int, int)> func);

  // Set the function that can calculate the number of output based on the
  // number of input. Use only one function in the set below.
  /**
   * @brief Set the output calculator to a user-defined function.
   */
  OpSchema& OutputCalculator(std::function<int(int)> calc);
  /**
   * @brief Set the number of outputs to be the same as the number of inputs.
   */
  OpSchema& SameNumberOfOutput();

  // Sets the rule to allow optional in-place operation.
  OpSchema& AllowConsumed(std::function<std::pair<bool,int>(int)> inplace);
  OpSchema& AllowConsumed(std::unordered_map<int, int> inplace);
  OpSchema& AllowOneToOneConsumed();
  // Sets the rule to enforce in-place opeartion.
  OpSchema& EnforceConsumed(std::function<std::pair<bool,int>(int)> inplace);
  OpSchema& EnforceConsumed(std::unordered_map<int, int> inplace);
  OpSchema& EnforceOneToOneConsumed();

  // Set the support level for the op schema.
  OpSchema& SetSupportLevel(SupportType supportType);

  // Functions to do documentation for the operator schema.
  OpSchema& SetDoc(const std::string& doc);

  // Note: this enum is structurally identical to the AttributeProto.AttributeType
  // enum defined in onnx.proto.  If you rev one, you likely need to rev the other.
  enum class AttrType {
      FLOAT,
      INT,
      STRING,
      TENSOR,
      GRAPH,
      FLOATS,
      INTS,
      STRINGS,
      TENSORS,
      GRAPHS
  };

  enum class UseType {
    DEFAULT, // read only use of an input
    CONSUME_ALLOWED, // allowed to be marked consumed by a "consumed_inputs" attribute.
    CONSUME_ENFORCED, // must be marked consumed by a "consumed_inputs" attribute.
  };

  struct Attribute {
    Attribute(const char* name_,
              const char* description_,
              AttrType type_,
              bool required_):
      name(name_),
      description(description_),
      type(type_),
      required(required_) {}

    const std::string name;
    const std::string description;
    AttrType type;
    bool required;
  };

  OpSchema& Attr(const Attribute& attr);
  OpSchema& Attr(const char* name,
                 const char* description,
                 AttrType type,
                 bool required = false);
  OpSchema& AllowUncheckedAttributes();

  // Optional = true means that the input might have empty input value
  // (represented as "") in the graph even though the later inputs have values.
  // It's useful for complex situation when there are several independent
  // optional inputs.
  OpSchema& Input(const int n, const char *name, const char *description,
                  bool optional = false);
  OpSchema& Output(const int n, const char *name, const char *description);
  // Calls the passed function with `this` as an argument. Useful for
  // adding docs for temlated/macro ops.
  OpSchema& FillUsing(std::function<void(OpSchema&)> populator);


  // Verifies that the schema is valid and all specifications are compatible.
  void Finalize();

  /**
   * @brief A function to allow one to get the number of outputs based on the
   * number of inputs, if this schema supports it.
   */
  int CalculateOutput(int num_input) const;

  friend std::ostream& operator<<(std::ostream& out, const OpSchema& schema);

  const std::map<std::string, Attribute>& attributes() const {
    return attributes_;
  }
  const std::vector<std::pair<const char*, const char*>>& input_desc() const {
    return input_desc_;
  }
  const std::vector<std::pair<const char*, const char*>>& output_desc() const {
    return output_desc_;
  }
  const std::set<int> optional_inputs() const {
    return optional_inputs_;
  }
  int min_input() const {
    return min_input_;
  }
  int max_input() const {
    return max_input_;
  }
  int min_output() const {
    return min_output_;
  }
  int max_output() const {
    return max_output_;
  }
  std::pair<UseType, int> consumed(int i) const {
    return consumed_(i);
  }

 private:
  std::string name_;
  std::string file_;
  std::string doc_;
  std::map<std::string, Attribute> attributes_{};
  bool allows_unchecked_attributes_ = false;
  std::vector<std::pair<const char*, const char*>> input_desc_{};
  std::vector<std::pair<const char*, const char*>> output_desc_{};
  int line_ = 0;
  SupportType support_;
  int min_input_ = 0;
  int max_input_ = std::numeric_limits<int>::max();
  int min_output_ = 0;
  int max_output_ = std::numeric_limits<int>::max();
  std::set<int> optional_inputs_;
  std::function<bool(int)> num_inputs_allowed_
      = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_
      = [](int) { return true; };
  std::function<bool(int, int)> num_inputs_outputs_allowed_
      = [](int, int) { return true; };
  std::function<int(int)> calculate_output_;
  // Is input i allowed/required to be marked consumed_
  // If so, which output idx shares the same buffer with i
  std::function<std::pair<UseType,int>(int)> consumed_
      = [](int){ return std::make_pair(UseType::DEFAULT, 0); };
};

/**
 * Internal class used in schema declaration
 */
class OpSchemaHolder {
 public:
  OpSchemaHolder(OpSchema& schema) : schema_(&schema) {
    // TODO: when we fix all issues - we can add abort() here
    try {
      schema.Finalize();
    } catch (const std::exception& e) {
      std::cerr << "Schema error: " << e.what() << std::endl;
    }
  }
  const OpSchema* operator->() const {
    return schema_;
  }

 private:
  const OpSchema* schema_;
};

/**
 * @brief A registry to hold all the operator schemas.
 */
class OpSchemaRegistry {
 public:
  static OpSchema& NewSchema(
    const std::string& key, const std::string& file, const int line) {
    auto& m = map();
    if (m.count(key)) {
      const auto& schema = m[key];
      std::cerr << "Trying to register schema with name "
                << key << " from file " << file << " line " << line
                << ", but it is already registered from file "
                << schema.file() << " line " << schema.line();
      abort();
    }
    m.emplace(std::make_pair(key, OpSchema(key, file, line)));
    return m[key];
  }

  static const OpSchema* Schema(const std::string& key) {
    auto& m = map();
    if (m.count(key)) {
      return &m[key];
    } else {
      return nullptr;
    }
  }

 private:
  // OpSchemaRegistry should not need to be instantiated.
  OpSchemaRegistry() = delete;

  /**
   * @brief Returns the underlying string to OpSchema map.
   *
   * You should not manually manipulate the map object returned. Instead, use
   * the macros defined such as OPERATOR_SCHEMA to register your operator
   * schema.
   *
   * We wrap it inside a function to avoid the statia initialization order
   * fiasco.
   */
  static std::unordered_map<std::string, OpSchema>& map();
 public:
  static const std::unordered_map<std::string, OpSchema>& registered_schemas() {
    return map();
  }
};

#define OPERATOR_SCHEMA(name)                                       \
  static onnx::OpSchemaHolder (op_schema_##name) =                     \
    onnx::OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

// Helper function
size_t ReplaceAll(std::string& s, const char* from, const char* to);

}  // namespace onnx
