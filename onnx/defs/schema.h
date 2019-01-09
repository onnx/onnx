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
#include "onnx/common/constants.h"
#include "onnx/defs/shape_inference.h"
namespace ONNX_NAMESPACE {

class SchemaError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  SchemaError(const std::string& message) : std::runtime_error(message) {}

  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string& context) {
    expanded_message_ = ONNX_NAMESPACE::MakeString(
        std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define fail_schema(...) \
  throw ONNX_NAMESPACE::SchemaError(ONNX_NAMESPACE::MakeString(__VA_ARGS__));

using OperatorSetVersion = int;

using DataTypeSet = std::unordered_set<DataType>;

// Type constraint map. Key is type string. Value is data type set and
// description.
using TypeConstraintMap =
    std::unordered_map<std::string, std::pair<DataTypeSet, std::string>>;

/**
 * @brief A class to record the schema of an op.
 *
 * OpSchema records the common interface of an op specified by its name.
 *
 * To register an OpSchema, one can use the macro ONNX_OPERATOR_SCHEMA(name) and
 * then append the various functions in the class. For example, for an op
 * that takes in two inputs, one output, and the first input and output
 * could be in-place, can be written as
 *
 *     ONNX_OPERATOR_SCHEMA(name)
 *         .NumInputs(2).NumOutputs(1).AllowConsumed({{0, 0}});
 *
 * To manufacture methods that may be used to register an OpSchema
 * non-statically, the following may be used:
 *
 *     ONNX_OPERATOR_SET_SCHEMA(name, version, OpSchema()
 *         .NumInputs(2).NumOutputs(1).AllowConsumed({{0, 0}}));
 */
class OpSchema final {
 public:
  // Formal parameter options.
  enum FormalParameterOption : uint8_t {
    // The input formal parameter is single and not optional.
    // Number of this input is 1.
    Single = 0,
    // The input formal parameter is single and optional.
    // Number of this input is 0 or 1.
    Optional = 1,
    // The input formal parameter is variadic.
    // Number of this input is [1, n].
    Variadic = 2,
  };

  // Formal parameter represenation, including input/output name, typeStr,
  // description, and type constraints.
  class FormalParameter final {
   public:
    // Constructor.
    FormalParameter() = default;

    explicit FormalParameter(
        std::string name,
        DataTypeSet type_set,
        std::string type_str,
        std::string description,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true);

    explicit FormalParameter(
        std::string name,
        std::string description,
        std::string type_str,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true);

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

    // Get whether a variadic parameter requires all to be of same type
    bool GetIsHomogeneous() const;

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

    // For variadic parameters, a flag indicating if all parameters must be of
    // same type
    bool is_homogeneous_;
  };

  enum class SupportType : uint8_t {
    COMMON, // Supported by all frameworks that support this IR.
    EXPERIMENTAL, // This OP is experimental and can be changed or removed in
                  // the future.
  };

  OpSchema() : OpSchema("unknown", "unknown", 0) {}
  OpSchema(std::string name, std::string file, int line)
      : name_(std::move(name)),
        file_(std::move(file)),
        line_(line),
        support_(SupportType::COMMON) {}

  /**
   * @brief Returns the file that the op schema is registered from.
   */
  const std::string& file() const {
    return file_;
  }

  /**
   * @brief Returns the line in file that the op schema is registered from.
   */
  int line() const {
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
  const char* doc() const {
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
   * Marks this op as deprecated as of it's since_version. This will cause the
   * Schema() lookup functions to return nullptr when the version is in the
   * deprecated range.
   */
  OpSchema& Deprecate();

  bool Deprecated() const {
    return deprecated_;
  }

  /**
   * @brief Input could be one of the values specified in allowed_input_nums.
   */
  OpSchema& NumInputs(std::set<int> allowed_input_nums);

  /**
   * @brief Output could be one of the values specified in allowed_output_nums.
   */
  OpSchema& NumOutputs(std::set<int> allowed_output_nums);

  // Shape Inference
  //
  // Note that signatures are defined to allow for forward-declaring
  // any structs used from ir.h
  OpSchema& TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction);
  InferenceFunction GetTypeAndShapeInferenceFunction() const {
    return tensor_inference_function_ ? tensor_inference_function_
                                      : dummyInferenceFunction;
  }

  // Set the support level for the op schema.
  OpSchema& SetSupportLevel(SupportType supportType);

  // Functions to do documentation for the operator schema.
  // This may be disabled to save memory.
  OpSchema& SetDoc(const char* doc) {
#ifndef __ONNX_NO_DOC_STRINGS
    SetDoc(std::string(doc));
#else
    doc;
#endif

    return *this;
  }

  OpSchema& SetDoc(std::string doc);

  // Functions to specify name for the operator schema.
  OpSchema& SetName(const char* name);
  OpSchema& SetName(std::string name);

  // Functions to specify code location for the operator schema.
  OpSchema& SetLocation(const char* file, int line);
  OpSchema& SetLocation(std::string file, int line);

  // Functions to specify domain for the operator schema.
  // Default domain value (ONNX_DOMAIN) means it's ONNX domain.
  OpSchema& SetDomain(const char* domain);
  OpSchema& SetDomain(std::string domain);

  struct Attribute final {
    Attribute(
        std::string name_,
        std::string description_,
        AttributeProto::AttributeType type_,
        bool required_)
        : name(std::move(name_)),
          description(std::move(description_)),
          type(type_),
          required(required_),
          default_value() {}

    Attribute(
        std::string name_,
        std::string description_,
        AttributeProto default_value_)
        : name(std::move(name_)),
          description(std::move(description_)),
          type(default_value_.type()),
          required(false),
          default_value(std::move(default_value_)) {}

    const std::string name;
    const std::string description;
    AttributeProto::AttributeType type;
    bool required;
    AttributeProto default_value;
  };

  OpSchema& Attr(Attribute attr);

// Register "optional" attribute with default value.
#define ATTR_SETTER_WITH_DEFAULT_VALUE(TypeName) \
  OpSchema& Attr(                                \
      std::string name,                          \
      std::string description,                   \
      AttributeProto::AttributeType type,        \
      const TypeName& defaultValue);             \
  /* non-STL wrapper to reduce binary size */    \
  OpSchema& Attr(                                \
      const char* name,                          \
      const char* description,                   \
      AttributeProto::AttributeType type,        \
      const TypeName& defaultValue);             \
  OpSchema& Attr(                                \
      std::string name,                          \
      std::string description,                   \
      AttributeProto::AttributeType type,        \
      const std::vector<TypeName>& defaultValue);

  ATTR_SETTER_WITH_DEFAULT_VALUE(int64_t)
  ATTR_SETTER_WITH_DEFAULT_VALUE(float)
  ATTR_SETTER_WITH_DEFAULT_VALUE(std::string)
  ATTR_SETTER_WITH_DEFAULT_VALUE(TensorProto)
  ATTR_SETTER_WITH_DEFAULT_VALUE(GraphProto)

  // Register "required" attribute without default value.
  OpSchema& Attr(
      std::string name,
      std::string description,
      AttributeProto::AttributeType type,
      bool required = true);

  // Non-STL wrapper to reduce binary size
  OpSchema& Attr(
      const char* name,
      const char* description,
      AttributeProto::AttributeType type,
      bool required = true);

  OpSchema& AllowUncheckedAttributes();

  // Type constraint.
  struct TypeConstraintParam final {
    TypeConstraintParam(
        std::string type_param_str_,
        std::vector<std::string> allowed_type_strs_,
        std::string description_)
        : type_param_str(std::move(type_param_str_)),
          allowed_type_strs(std::move(allowed_type_strs_)),
          description(std::move(description_)) {}

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
  // ONNX_OPERATOR_SET_SCHEMA(Sum, 1, OpSchema()
  // .Input(0, "input_a", "the first input", "T")
  // .Input(1, "input_b", "the second input", "T")
  // .Output(0, "sum", "the sum of two numbers", "T")
  // .TypeConstraint("T", {"float", "double", "int32"}, "allowed data types for
  // sum."))
  //
  // Optional = true means that the input might have empty input value
  // (represented as "") in the graph even though the later inputs have values.
  // It's useful for complex situation when there are several independent
  // optional inputs.
  OpSchema& Input(
      int n,
      std::string name,
      std::string description,
      std::string type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true);

  // Non-STL wrapper to reduce binary size
  OpSchema& Input(
      int n,
      const char* name,
      const char* description,
      const char* type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true);

  OpSchema& Output(
      int n,
      std::string name,
      std::string description,
      std::string type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true);

  // Non-STL wrapper to reduce binary size
  OpSchema& Output(
      int n,
      const char* name,
      const char* description,
      const char* type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true);

  OpSchema& TypeConstraint(
      std::string type_str,
      std::vector<std::string> constraints,
      std::string description);

  // Non-STL wrapper to reduce binary size
  OpSchema& TypeConstraint(
      const char* type_str,
      std::initializer_list<const char*> constraints,
      const char* description);

  // Convenience members for types

  // All high-precision numeric types.
  static const std::vector<std::string>& numeric_types_for_math_reduction() {
    static const std::vector<std::string> numeric_types_for_math_reduction = {
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)"};
    return numeric_types_for_math_reduction;
  }

  static const std::vector<std::string>& all_numeric_types() {
    static const std::vector<std::string> all_numeric_types = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)"};
    return all_numeric_types;
  }

  static const std::vector<std::string>& all_tensor_types() {
    static const std::vector<std::string> all_tensor_types = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)",
        "tensor(bool)",
        "tensor(complex64)",
        "tensor(complex128)"};
    return all_tensor_types;
  }

  // Calls the passed function with `this` as an argument. Useful for
  // adding docs for temlated/macro ops.
  OpSchema& FillUsing(const std::function<void(OpSchema&)>& populator);

  friend std::ostream& operator<<(std::ostream& out, const OpSchema& schema);

  const std::string& domain() const {
    return domain_;
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

  OperatorSetVersion SinceVersion() const {
    return since_version_;
  }

  int since_version() const {
    return since_version_;
  }

  bool deprecated() const {
    return deprecated_;
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

  bool has_type_and_shape_inference_function() const {
    return tensor_inference_function_ ? true : false;
  }

  // Verifies that the schema is valid and all specifications are compatible.
  // It will also parse all type strings specified for inputs/outputs into valid
  // TypeProto and create global unique string pointer as the DataType for
  // efficiency.
  void Finalize();

 private:
  void ParseAndSetTypes(
      /*out*/ std::vector<OpSchema::FormalParameter>* formalParameters);

  std::string name_;
  std::string file_;
  std::string doc_;
  // Default domain value ("") means it's ONNX domain.
  std::string domain_ = ONNX_DOMAIN;
  std::map<std::string, Attribute> attributes_{};
  bool allows_unchecked_attributes_ = false;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraint_params_;
  TypeConstraintMap type_constraints_;
  int line_ = 0;
  SupportType support_;
  int min_input_ = 0;
  int max_input_ = 0;
  int min_output_ = 0;
  int max_output_ = 0;
  // The default is a little goofy, since it is never what you want
  OperatorSetVersion since_version_ = 1;
  bool deprecated_{};
  std::function<bool(int)> num_inputs_allowed_ = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_ = [](int) { return true; };
  InferenceFunction tensor_inference_function_;
};

// Map type to store operator schemas. The format is,
// <OpName, <Domain, <OperatorSetVersion, OpSchema>>>.
using OpName_Domain_Version_Schema_Map = std::unordered_map<
    std::string,
    std::unordered_map<std::string, std::map<OperatorSetVersion, OpSchema>>>;

class ISchemaRegistry {
 public:
  virtual ~ISchemaRegistry() = default;

  virtual const OpSchema* GetSchema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = ONNX_DOMAIN) const = 0;
};

/**
 * @brief A registry to hold all the operator schemas.
 */
class OpSchemaRegistry final : public ISchemaRegistry {
 public:
  // A singleton class to store domain to min/max op_set version map.
  class DomainToVersionRange final {
   public:
    DomainToVersionRange() {
      // Increase the highest version when you make BC-breaking changes to the
      // operator schema on specific domain. Update the lowest version when it's
      // determined to remove too old version history.
      map_[ONNX_DOMAIN] = std::make_pair(1, 9);
      map_[AI_ONNX_ML_DOMAIN] = std::make_pair(1, 2);
    }

    const std::unordered_map<std::string, std::pair<int, int>>& Map() const {
      return map_;
    }

    // Add customized domain to min/max version.
    // Onnx partners are able to use onnx operator schema api to
    // register customized op in their own domain.
    void AddDomainToVersion(
        const std::string& domain,
        int min_version,
        int max_version) {
      std::lock_guard<std::mutex> lock(mutex_);
      assert(map_.end() == map_.find(domain));
      map_[domain] = std::make_pair(min_version, max_version);
    }

    static DomainToVersionRange& Instance();

   private:
    // Key: domain. Value: <lowest version, highest version> pair.
    std::unordered_map<std::string, std::pair<int, int>> map_;

    std::mutex mutex_;
  };

  class OpSchemaRegisterOnce final {
   public:
    OpSchemaRegisterOnce(OpSchema& op_schema) {
      try {
        op_schema.Finalize();

        auto& m = GetMapWithoutEnsuringRegistration();

        auto& op_name = op_schema.Name();
        auto& op_domain = op_schema.domain();
        auto ver = op_schema.SinceVersion();

        if (m[op_name][op_domain].count(ver)) {
          const auto& schema = m[op_name][op_domain][ver];
          std::stringstream err;
          err << "Trying to register schema with name " << op_name
              << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line "
              << op_schema.line() << ", but it is already registered from file "
              << schema.file() << " line " << schema.line() << std::endl;
          fail_schema(err.str());
        }

        auto ver_range_map = DomainToVersionRange::Instance().Map();
        auto ver_range_it = ver_range_map.find(op_domain);
        if (ver_range_it == ver_range_map.end()) {
          std::stringstream err;
          err << "Trying to register schema with name " << op_name
              << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line "
              << op_schema.line() << ", but it its domain is not"
              << "known by the checker." << std::endl;

          fail_schema(err.str());
        }
        auto lower_bound_incl = ver_range_it->second.first;
        auto upper_bound_incl = ver_range_it->second.second;
        if (!(lower_bound_incl <= ver && upper_bound_incl >= ver)) {
          std::stringstream err;
          err << "Trying to register schema with name " << op_name
              << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line "
              << op_schema.line() << ", but it its version is not"
              << "in the inclusive range [" << lower_bound_incl << ", "
              << upper_bound_incl << "] (usually, this means you "
              << "bumped the operator version but "
              << "forgot to update the version range in DomainToVersionRange "
              << "in onnx/defs/schema.h)." << std::endl;
          fail_schema(err.str());
        }
        m[op_name][op_domain].emplace(std::make_pair(ver, op_schema));

      } catch (const std::exception& e) {
        std::cerr << "Schema error: " << e.what() << std::endl;
      }
    }
  };

  // Return the latest schema for an operator in specified domain.
  // Domain with default value ONNX_DOMAIN means ONNX.
  static const OpSchema* Schema(
      const std::string& key,
      const std::string& domain = ONNX_DOMAIN) {
    auto& m = map();
    if (m.count(key) && m[key].count(domain)) {
      return &m[key][domain].rbegin()->second;
    } else {
      return nullptr;
    }
  }

  // Return the schema with biggest version, which is not greater than specified
  // <maxInclusiveVersion> in specified domain. Domain with default value
  // ONNX_DOMAIN means ONNX.
  static const OpSchema* Schema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = ONNX_DOMAIN) {
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
      }

      // Schema with exact version as specified one exists.
      return &(pos->second);
    } else {
      return nullptr;
    }
  }

  static OpSchemaRegistry* Instance();

  const OpSchema* GetSchema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = ONNX_DOMAIN) const override {
    return Schema(key, maxInclusiveVersion, domain);
  }

 private:
  // OpSchemaRegistry should not need to be instantiated except statically
  // within this class
  OpSchemaRegistry() = default;

  /**
   * @brief Returns the underlying string to OpSchema map.
   *
   * You should not manually manipulate the map object returned. Instead, use
   * the macros defined such as ONNX_OPERATOR_SET_SCHEMA to register your
   * operator schema.
   *
   * We wrap it inside a function to avoid the statia initialization order
   * fiasco.
   */
  static OpName_Domain_Version_Schema_Map& GetMapWithoutEnsuringRegistration();
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

void RegisterSchema(OpSchema&& schema);

// Registers all schema of a given operator set
template <class T>
void RegisterOpSetSchema() {
  T::ForEachSchema(RegisterSchema);
};

// Forward declaration for the non-specialized GetOpSchema method.  This
// enforces a consistent signature on functions that query individual schema,
// which are defined as specializations of this function.
template <typename T>
OpSchema GetOpSchema();

#define ONNX_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, Onnx, ONNX_DOMAIN, ver, true, impl)

#define ONNX_ML_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxML, AI_ONNX_ML_DOMAIN, ver, true, impl)

// Defines specialization of GetOpSchema for a class whose name is determined
// based on a convention using name, domain, and version.  Operator schema are
// normally included in operator sets and registered in OpSchemaRegistry::map().
// In this case, callers should set dbg_included_in_static_opset to true.  This
// assists with runtime validation in in DEBUG builds ensuring the intended set
// of operator schema is registered.
#define ONNX_OPERATOR_SET_SCHEMA_EX(                                        \
    name, domain, domain_str, ver, dbg_included_in_static_opset, impl)      \
  class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name);             \
  template <>                                                               \
  OpSchema                                                                  \
  GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)>() {   \
    return impl.SetName(#name)                                              \
        .SetDomain(domain_str)                                              \
        .SinceVersion(ver)                                                  \
        .SetLocation(__FILE__, __LINE__);                                   \
  }                                                                         \
  size_t dbg_count_check_##name##_##domain##_ver##ver =                     \
      (dbg_included_in_static_opset) ? ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() \
                                     : 0;

#ifdef NDEBUG
#define ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() 0
#else
#define ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() \
  DbgOperatorSetTracker::Instance().IncrementCount()
#define ONNX_DBG_GET_COUNT_IN_OPSETS() \
  DbgOperatorSetTracker::Instance().GetCount()

class DbgOperatorSetTracker {
 public:
  static DbgOperatorSetTracker& Instance();

  size_t IncrementCount() {
    return ++count_;
  }

  size_t GetCount() const {
    return count_;
  }

 private:
  size_t count_ = 0;
};
#endif

// Naming convention for operator schema classes
#define ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name) \
  name##_##domain##_ver##ver

// Helper function
size_t ReplaceAll(std::string& s, const char* from, const char* to);

#ifdef __GNUC__
#define ONNX_UNUSED __attribute__((__unused__))
#else
#define ONNX_UNUSED
#endif

// Legacy macros to register schema at static initialization
#define ONNX_OPERATOR_SCHEMA(name) \
  ONNX_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_OPERATOR_SCHEMA_UNIQ(Counter, name)                 \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      OpSchema(#name, __FILE__, __LINE__)

// Helper function
size_t ReplaceAll(std::string& s, const char* from, const char* to);

inline std::string GenerateOptionalArgumentsDoc() {
  return "This operator has **optional** inputs/outputs. "
         "See [the doc](IR.md) for more details about the representation of "
         "optional arguments. An empty string may be used in the place of "
         "an actual argument's name to indicate a missing argument. "
         "Trailing optional arguments (those not followed by an argument "
         "that is present) may also be simply omitted.\n";
}

inline std::string GenerateBroadcastingDocMul() {
  return "This operator supports **multidirectional (i.e., Numpy-style) broadcasting**;"
         " for more details please check [the doc](Broadcasting.md).";
}

inline std::string GenerateBroadcastingDocUni(
    const char* from,
    const char* to) {
  std::string ret = "This operator supports **unidirectional broadcasting** (";
  ret = ret + from + " should be unidirectional broadcastable to " + to +
      ");"
      " for more details please check [the doc](Broadcasting.md).";
  return ret;
}

} // namespace ONNX_NAMESPACE
