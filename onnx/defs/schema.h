// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <climits>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "data_type_utils.h"

namespace onnx {

using OperatorSetVersion = int;

typedef std::unordered_set<DataType> DataTypeSet;

// Type constraint map. Key is type string. Value is data type set and
// description.
typedef std::unordered_map<std::string, std::pair<DataTypeSet, std::string>>
    TypeConstraintMap;

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
  // Formal parameter options.
  enum FormalParameterOption {
    // The input formal parameter is single and not optional.
    // Number of this input is 1.
    Single = 0,
    // The input formal parameter is single and optional.
    // Number of this input is 0 or 1.
    Optional = 1,
    // The input formal parameter is variadic.
    // Number of this input is [0, n].
    Variadic = 2,
  };

  // Formal parameter represenation, including input/output name, typeStr,
  // description, and type constraints.
  class FormalParameter {
   public:
    // Constructor.
    FormalParameter() = default;

    explicit FormalParameter(
        const std::string& name,
        const DataTypeSet& type_set,
        const std::string& type_str,
        const std::string& description,
        FormalParameterOption param_option = Single);

    explicit FormalParameter(
        const std::string& name,
        const std::string& description,
        const std::string& type_str,
        FormalParameterOption param_option = Single);

    // Get formal parameter name.
    const std::string& GetName() const;

    // Get allowed data types.
    const DataTypeSet& GetTypes() const;

    // Get formal parameter type string.
    const std::string& GetTypeStr() const;

    // Get formal parameter description.
    const std::string& GetDescription() const;

    // Get the parameter option, it could be Single, Optional or Variadic.
    FormalParameterOption GetOption() const;

   private:
    friend class OpSchema;

    DataTypeSet& MutableTypes();

    // Formal parameter name.
    std::string name_;

    // A set of data types supported for <*this> formal parameter.
    // It should contain at least one element if this formal parameter is good.
    DataTypeSet type_set_;

    // The <parameter type> string specified when registring an op.
    // It could be a supported data type or a type constraint key, which
    // maps to a set of supported data types.
    std::string type_str_;

    // Formal parameter description.
    std::string description_;

    // Formal parameter option.
    FormalParameterOption param_option_;
  };

  enum class SupportType {
    COMMON, // Supported by all frameworks that support this IR.
    EXPERIMENTAL, // This OP is experimental and can be changed or removed in
                  // the future.
  };

  OpSchema()
      : name_("unknown"),
        file_("unknown"),
        line_(0),
        support_(SupportType::COMMON) {}
  OpSchema(const std::string& name, const std::string& file, const int line)
      : name_(name), file_(file), line_(line), support_(SupportType::COMMON) {}

  /**
   * @brief Returns the file that the op schema is registered from.
   */
  inline const std::string& file() const {
    return file_;
  }

  /**
   * @brief Returns the line in file that the op schema is registered from.
   */
  inline int line() const {
    return line_;
  }

  /**
   * @brief Returns the support level of the op schema.
   */
  SupportType support_level() const {
    return support_;
  }

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
  void Verify(const NodeProto& node) const;

  // Functions to set the property of the operator schemas.
  // Sets the number of inputs, either a fixed number or a min and a max.

  /**
   * The earliest operator set version which this operator was
   * present in.  If an operator has had no BC-breaking changes,
   * this is simply the first operator set the operator was a member
   * of; if it has had BC-breaking changes, then for the semantics
   * /as described/ in the OpSchema entry, this version describes
   * the operator set which introduced the BC-breaking change.
   *
   * For example, suppose op Foo was added in v3, and had a BC-breaking
   * change in v6.  Then there will be an op schema entry for Foo with
   * SinceVersion(3), and another, updated op schema entry for Foo
   * with SinceVersion(6).
   */
  OpSchema& SinceVersion(OperatorSetVersion n); // aka int

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
  OpSchema& AllowConsumed(std::function<std::pair<bool, int>(int)> inplace);
  OpSchema& AllowConsumed(std::unordered_map<int, int> inplace);
  OpSchema& AllowOneToOneConsumed();
  // Sets the rule to enforce in-place opeartion.
  OpSchema& EnforceConsumed(std::function<std::pair<bool, int>(int)> inplace);
  OpSchema& EnforceConsumed(std::unordered_map<int, int> inplace);
  OpSchema& EnforceOneToOneConsumed();

  // Set the support level for the op schema.
  OpSchema& SetSupportLevel(SupportType supportType);

  // Functions to do documentation for the operator schema.
  OpSchema& SetDoc(const std::string& doc);

  // Functions to specify domain for the operator schema.
  // Default domain value ("") means it's ONNX domain.
  OpSchema& SetDomain(const std::string& domain);

  // Note: this enum is structurally identical to the
  // AttributeProto.AttributeType enum defined in onnx.proto.  If you rev one,
  // you likely need to rev the other.
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
    CONSUME_ALLOWED, // allowed to be marked consumed by a "consumed_inputs"
                     // attribute.
    CONSUME_ENFORCED, // must be marked consumed by a "consumed_inputs"
                      // attribute.
  };

  struct Attribute {
    Attribute(
        const char* name_,
        const char* description_,
        AttrType type_,
        bool required_)
        : name(name_),
          description(description_),
          type(type_),
          required(required_) {}

    const std::string name;
    const std::string description;
    AttrType type;
    bool required;
  };

  OpSchema& Attr(const Attribute& attr);
  OpSchema& Attr(
      const char* name,
      const char* description,
      AttrType type,
      bool required = false);
  OpSchema& AllowUncheckedAttributes();

  // Type constraint.
  struct TypeConstraintParam {
    TypeConstraintParam(
        const std::string& type_param_str_,
        const std::vector<std::string>& allowed_type_strs_,
        const std::string& description_)
        : type_param_str(type_param_str_),
          allowed_type_strs(allowed_type_strs_),
          description(description_) {}

    // Type parameter string, for example, "T", "T1", etc.
    std::string type_param_str;
    // Allowed type strings for <*this> type parameter, for example,
    // "tensor(float)".
    std::vector<std::string> allowed_type_strs;
    // Type parameter description.
    std::string description;
  };

  // Grammar for type strings used in Input(), Output().
  // <type> ::= <data_type> |
  //            tensor(<data_type>) |
  //            seq(<type>) |
  //            map(<data_type>, <type>) |
  //            <type_parameter>
  // <data_type> :: = float | int32 | string | bool | uint8
  //                | int8 | uint16 | int16 | int64 | float16 | double
  // <type_parameter> ::= any type parameter string, say "T".
  //
  // NOTE: 1) <type_parameter> will always be together with a type constraints
  // specification.
  //       2) <type> ::= <data_type> means the data is scalar (zero dimension).
  //
  // Example:
  // OPERATOR_SCHEMA(Sum)
  // .Input(0, "input_a", "the first input", "T")
  // .Input(1, "input_b", "the second input", "T")
  // .Output(0, "sum", "the sum of two numbers", "T")
  // .TypeConstraint("T", {"float", "double", "int32"}, "allowed data types for
  // sum.")
  //
  // Optional = true means that the input might have empty input value
  // (represented as "") in the graph even though the later inputs have values.
  // It's useful for complex situation when there are several independent
  // optional inputs.
  OpSchema& Input(
      const int n,
      const std::string& name,
      const std::string& description,
      const std::string& type_str,
      FormalParameterOption param_option = Single);
  OpSchema& Output(
      const int n,
      const std::string& name,
      const std::string& description,
      const std::string& type_str);
  OpSchema& TypeConstraint(
      const std::string& type_str,
      const std::vector<std::string>& constraints,
      const std::string& description);

  // Calls the passed function with `this` as an argument. Useful for
  // adding docs for temlated/macro ops.
  OpSchema& FillUsing(std::function<void(OpSchema&)> populator);

  /**
   * @brief A function to allow one to get the number of outputs based on the
   * number of inputs, if this schema supports it.
   */
  int CalculateOutput(int num_input) const;

  friend std::ostream& operator<<(std::ostream& out, const OpSchema& schema);

  const std::string& domain() const {
    return domain_;
  }

  int since_version() const {
    return since_version_;
  }
  const std::map<std::string, Attribute>& attributes() const {
    return attributes_;
  }

  // Get input formal parameters.
  const std::vector<FormalParameter>& inputs() const {
    return inputs_;
  }

  // Get output formal parameters.
  const std::vector<FormalParameter>& outputs() const {
    return outputs_;
  }

  const std::vector<TypeConstraintParam>& typeConstraintParams() const {
    return type_constraint_params_;
  }

  const std::string& Name() const {
    return name_;
  }

  const OperatorSetVersion SinceVersion() const {
    return since_version_;
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
  friend class OpSchemaRegistry;

  // Verifies that the schema is valid and all specifications are compatible.
  // It will also parse all type strings specified for inputs/outputs into valid
  // TypeProto and create global unique string pointer as the DataType for
  // efficiency.
  void Finalize();

  void ParseAndSetTypes(
      /*out*/ std::vector<OpSchema::FormalParameter>* formalParameters);

  std::string name_;
  std::string file_;
  std::string doc_;
  // Default domain value ("") means it's ONNX domain.
  std::string domain_ = "";
  std::map<std::string, Attribute> attributes_{};
  bool allows_unchecked_attributes_ = false;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraint_params_;
  TypeConstraintMap type_constraints_;
  int line_ = 0;
  SupportType support_;
  int min_input_ = 0;
  int max_input_ = std::numeric_limits<int>::max();
  int min_output_ = 0;
  int max_output_ = std::numeric_limits<int>::max();
  // The default is a little goofy, since it is never what you want
  OperatorSetVersion since_version_ = 1;
  std::function<bool(int)> num_inputs_allowed_ = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_ = [](int) { return true; };
  std::function<bool(int, int)> num_inputs_outputs_allowed_ = [](int, int) {
    return true;
  };
  std::function<int(int)> calculate_output_;
  // Is input i allowed/required to be marked consumed_
  // If so, which output idx shares the same buffer with i
  std::function<std::pair<UseType, int>(int)> consumed_ = [](int) {
    return std::make_pair(UseType::DEFAULT, 0);
  };
};

// Map type to store operator schemas. The format is,
// <OpName, <Domain, <OperatorSetVersion, OpSchema>>>.
typedef std::unordered_map<
    std::string,
    std::unordered_map<std::string, std::map<OperatorSetVersion, OpSchema>>>
    OpName_Domain_Version_Schema_Map;    

/**
 * @brief A registry to hold all the operator schemas.
 */
class OpSchemaRegistry {
 public:
  class DomainToVersionRange {
   public:
    DomainToVersionRange() {
      // Increase the highest version when you make BC-breaking changes to the
      // operator schema on specific domain. Update the lowest version when it's
      // determined to remove too old version history.
      map_[""] = std::make_pair(1, 2);
    }

    const std::unordered_map<std::string, std::pair<int, int>>& Map() const {
      return map_;
    }

    static DomainToVersionRange& Instance() {
      static DomainToVersionRange domain_to_version_range;
      return domain_to_version_range;
    }

   private:
    // Key: domain. Value: <lowest version, highest version> pair.
    std::unordered_map<std::string, std::pair<int, int>> map_;
  };

  class OpSchemaRegisterOnce {
   public:
    OpSchemaRegisterOnce(OpSchema& op_schema) {
      // TODO: when we fix all issues - we can add abort() here
      try {
        op_schema.Finalize();
      } catch (const std::exception& e) {
        std::cerr << "Schema error: " << e.what() << std::endl;
      }
      auto& m = map();
      auto& op_name = op_schema.Name();
      auto& op_domain = op_schema.domain();
      auto ver = op_schema.SinceVersion();

      if (m[op_name][op_domain].count(ver)) {
        const auto& schema = m[op_name][op_domain][ver];
        std::cerr << "Trying to register schema with name " << op_name
                  << " (domain: " << op_domain << " version: " << ver
                  << ") from file " << op_schema.file() << " line "
                  << op_schema.line()
                  << ", but it is already registered from file "
                  << schema.file() << " line " << schema.line();
        abort();
      }
      m[op_name][op_domain].emplace(std::make_pair(ver, op_schema));
    }
  };

  // Return the latest schema for an operator in specified domain.
  // Domain with default value "" means ONNX.
  static const OpSchema* Schema(
      const std::string& key,
      const std::string& domain = "") {
    auto& m = map();
    if (m.count(key) && m[key].count(domain)) {
      return &m[key][domain].rbegin()->second;
    } else {
      return nullptr;
    }
  }

  // Return the schema with biggest version, which is not greater than specified
  // <maxInclusiveVersion> in specified domain. Domain with default value "" means ONNX.
  static const OpSchema* Schema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = "") {
    auto& m = map();
    if (m.count(key) && m[key].count(domain)) {
      auto pos = m[key][domain].lower_bound(maxInclusiveVersion);
      if (m[key][domain].begin() == pos && pos->first > maxInclusiveVersion) {
        // All versions are greater than specified version.
        return nullptr;
      }
      if (m[key][domain].end() == pos || pos->first > maxInclusiveVersion) {
        // All versions are less than specified version, or,
        // The <pos> version is greater than specified version.
        pos--;
        return &(pos->second);
      }
      // Schema with exact version as specified one exists.
      return &(pos->second);
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
  static OpName_Domain_Version_Schema_Map& map();

 public:
  static const std::vector<OpSchema> get_all_schemas_with_history() {
    std::vector<OpSchema> r;
    for (auto x : map()) {
      for (auto y : x.second) {
        for (auto z : y.second) {
          r.emplace_back(z.second);
        }
      }
    }
    return r;
  }

  static const std::vector<OpSchema> get_all_schemas() {
    std::vector<OpSchema> r;
    for (auto x : map()) {
      for (auto y : x.second) {
        auto& version2schema = y.second;
        r.emplace_back(version2schema.rbegin()->second);
      }
    }
    return r;
  }
};

#define OPERATOR_SCHEMA(name)                          \
  static onnx::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name) = OpSchema(#name, __FILE__, __LINE__)

// Helper function
size_t ReplaceAll(std::string& s, const char* from, const char* to);
} // namespace onnx
