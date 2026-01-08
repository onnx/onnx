// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <climits>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx/checker.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/inliner/inliner.h"
#include "onnx/py_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

#ifdef ONNX_USE_LITE_PROTO
using BASE_PROTO_TYPE = ::google::protobuf::MessageLite;
#else
using BASE_PROTO_TYPE = ::google::protobuf::Message;
#endif

// Generic type caster template for ONNX Proto classes
#define ONNX_DEFINE_TYPE_CASTER(ProtoType, PythonClassName)                                       \
  template <>                                                                                     \
  struct nanobind::detail::type_caster<ONNX_NAMESPACE::ProtoType> {                               \
   public:                                                                                        \
    NB_TYPE_CASTER(ONNX_NAMESPACE::ProtoType, nanobind::detail::const_name(PythonClassName));     \
                                                                                                  \
    bool from_python(handle py_proto, uint8_t, cleanup_list*) noexcept {                          \
      try {                                                                                       \
        if (!nanobind::hasattr(py_proto, "SerializeToString")) {                                  \
          return false;                                                                           \
        }                                                                                         \
        auto serialized = nanobind::cast<nanobind::bytes>(py_proto.attr("SerializeToString")());  \
        if (!ParseProtoFromPyBytes(&value, serialized)) {                                         \
          return false;                                                                           \
        }                                                                                         \
        return true;                                                                              \
      } catch (const nanobind::python_error&) {                                                   \
        return false;                                                                             \
      }                                                                                           \
    }                                                                                             \
                                                                                                  \
    static handle from_cpp(                                                                       \
        const ONNX_NAMESPACE::ProtoType& cpp_proto,                                               \
        rv_policy /* policy */,                                                                   \
        cleanup_list* /* cleanup */) noexcept {                                                   \
      try {                                                                                       \
        std::string serialized = cpp_proto.SerializeAsString();                                   \
        auto py_proto = nanobind::module_::import_("onnx").attr(#ProtoType)();                    \
        py_proto.attr("ParseFromString")(nanobind::bytes(serialized.c_str(), serialized.size())); \
        return py_proto.release();                                                                \
      } catch (...) {                                                                             \
        return handle();                                                                          \
      }                                                                                           \
    }                                                                                             \
  };

// Define type casters for common ONNX proto types
ONNX_DEFINE_TYPE_CASTER(AttributeProto, "onnx.AttributeProto")
ONNX_DEFINE_TYPE_CASTER(TypeProto, "onnx.TypeProto")
ONNX_DEFINE_TYPE_CASTER(TensorProto, "onnx.TensorProto")
ONNX_DEFINE_TYPE_CASTER(SparseTensorProto, "onnx.SparseTensorProto")
ONNX_DEFINE_TYPE_CASTER(ValueInfoProto, "onnx.ValueInfoProto")
ONNX_DEFINE_TYPE_CASTER(NodeProto, "onnx.NodeProto")
ONNX_DEFINE_TYPE_CASTER(GraphProto, "onnx.GraphProto")
ONNX_DEFINE_TYPE_CASTER(ModelProto, "onnx.ModelProto")
ONNX_DEFINE_TYPE_CASTER(FunctionProto, "onnx.FunctionProto")

namespace ONNX_NAMESPACE {
namespace nb = nanobind;
using namespace nanobind::literals;

template <typename ProtoType>
static std::tuple<bool, nb::bytes, nb::bytes> Parse(const char* cstr) {
  ProtoType proto{};
  OnnxParser parser(cstr);
  auto status = parser.Parse(proto);
  std::string out;
  proto.SerializeToString(&out);
  std::string error_msg = status.ErrorMessage();
  return std::make_tuple(
      status.IsOK(), nb::bytes(error_msg.c_str(), error_msg.size()), nb::bytes(out.c_str(), out.size()));
}

template <typename ProtoType>
static std::string ProtoBytesToText(const nb::bytes& bytes) {
  ProtoType proto{};
  ParseProtoFromPyBytes(&proto, bytes);
  return ProtoToString(proto);
}

template <typename T, typename Ts = std::remove_const_t<T>>
static std::pair<std::vector<Ts>, std::unordered_map<std::string, T*>> ParseProtoFromBytesMap(
    const std::unordered_map<std::string, nb::bytes>& bytesMap) {
  std::vector<Ts> values(bytesMap.size());
  std::unordered_map<std::string, T*> result;
  size_t i = 0;
  for (const auto& kv : bytesMap) {
    ParseProtoFromPyBytes(&values[i], kv.second);
    result[kv.first] = &values[i];
    i++;
  }
  // C++ guarantees that the pointers remain valid after std::vector<Ts> is moved.
  return std::make_pair(std::move(values), result);
}

static std::unordered_map<std::string, nb::bytes> CallNodeInferenceFunction(
    OpSchema* schema,
    const nb::bytes& nodeBytes,
    const std::unordered_map<std::string, nb::bytes>& valueTypesByNameBytes,
    const std::unordered_map<std::string, nb::bytes>& inputDataByNameBytes,
    const std::unordered_map<std::string, nb::bytes>& inputSparseDataByNameBytes,
    std::unordered_map<std::string, int> opsetImports,
    const int irVersion) {
  NodeProto node{};
  ParseProtoFromPyBytes(&node, nodeBytes);
  // Early fail if node is badly defined - may throw ValidationError
  schema->Verify(node);

  // Convert arguments to C++ types, allocating memory
  const auto& valueTypes = ParseProtoFromBytesMap<TypeProto>(valueTypesByNameBytes);
  const auto& inputData = ParseProtoFromBytesMap<const TensorProto>(inputDataByNameBytes);
  const auto& inputSparseData = ParseProtoFromBytesMap<const SparseTensorProto>(inputSparseDataByNameBytes);
  if (opsetImports.empty()) {
    opsetImports[schema->domain()] = schema->SinceVersion();
  }

  shape_inference::GraphInferenceContext graphInferenceContext(
      valueTypes.second, opsetImports, nullptr, {}, OpSchemaRegistry::Instance(), nullptr, irVersion);
  // Construct inference context and get results - may throw InferenceError
  // TODO: if it is desirable for infer_node_outputs to provide check_type, strict_mode, data_prop,
  // we can add them to the Python API. For now we just assume the default options.
  ShapeInferenceOptions options{false, 0, false};
  shape_inference::InferenceContextImpl ctx(
      node, valueTypes.second, inputData.second, inputSparseData.second, options, nullptr, &graphInferenceContext);
  schema->GetTypeAndShapeInferenceFunction()(ctx);
  // Verify the inference succeeded - may also throw ValidationError
  // Note that input types were not validated until now (except that their count was correct)
  schema->CheckInputOutputType(ctx);

  // Convert back into bytes returned to Python
  std::unordered_map<std::string, nb::bytes> typeProtoBytes;
  for (size_t i = 0; i < ctx.allOutputTypes_.size(); i++) {
    const auto& proto = ctx.allOutputTypes_[i];
    if (proto.IsInitialized()) {
      std::string s;
      proto.SerializeToString(&s);
      typeProtoBytes[node.output(static_cast<int>(i))] = nb::bytes(s.c_str(), s.size());
    }
  }

  return typeProtoBytes;
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<const T*>> ConvertPyObjToPtr(const std::vector<nb::object>& pyObjs) {
  std::vector<T> objs;
  std::vector<const T*> ptrs;
  objs.reserve(pyObjs.size());
  ptrs.reserve(pyObjs.size());
  for (const auto& obj : pyObjs) {
    if (obj.is_none()) {
      ptrs.push_back(nullptr);
      continue;
    }
    objs.emplace_back(nanobind::cast<T>(obj));
    ptrs.push_back(&objs.back());
  }
  return std::make_tuple(std::move(objs), std::move(ptrs));
}

NB_MODULE(onnx_cpp2py_export, onnx_cpp2py_export) {
  // Disabling nanobind leak warnings
  // TODO(#7283): Avoid leaks if possible
  nb::set_leak_warnings(false);

  onnx_cpp2py_export.doc() = "Python interface to ONNX";

  onnx_cpp2py_export.attr("ONNX_ML") = nb::bool_(
#ifdef ONNX_ML
      true
#else // ONNX_ML
      false
#endif // ONNX_ML
  );

  // Avoid Segmentation fault if we not free the python function in Custom Schema
  // onnx_cpp2py_export.attr("_cleanup") = nb::capsule(+[] { OpSchemaRegistry::OpSchemaDeregisterAll(); });

  // Submodule `schema`
  auto defs = onnx_cpp2py_export.def_submodule("defs");
  defs.doc() = "Schema submodule";
  nb::exception<SchemaError>(defs, "SchemaError");

  nb::class_<OpSchema> op_schema(defs, "OpSchema", "Schema of an operator.");

  // Define the class enums first because they are used as default values in function definitions
  nb::enum_<OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption", nb::is_arithmetic())
      .value("Single", OpSchema::Single)
      .value("Optional", OpSchema::Optional)
      .value("Variadic", OpSchema::Variadic);

  nb::enum_<OpSchema::DifferentiationCategory>(op_schema, "DifferentiationCategory", nb::is_arithmetic())
      .value("Unknown", OpSchema::Unknown)
      .value("Differentiable", OpSchema::Differentiable)
      .value("NonDifferentiable", OpSchema::NonDifferentiable);

  nb::enum_<OpSchema::NodeDeterminism>(op_schema, "NodeDeterminism")
      .value("Deterministic", OpSchema::NodeDeterminism::Deterministic)
      .value("NonDeterministic", OpSchema::NodeDeterminism::NonDeterministic)
      .value("Unknown", OpSchema::NodeDeterminism::Unknown);

  nb::enum_<AttributeProto::AttributeType>(op_schema, "AttrType", nb::is_arithmetic())
      .value("FLOAT", AttributeProto::FLOAT)
      .value("INT", AttributeProto::INT)
      .value("STRING", AttributeProto::STRING)
      .value("TENSOR", AttributeProto::TENSOR)
      .value("GRAPH", AttributeProto::GRAPH)
      .value("FLOATS", AttributeProto::FLOATS)
      .value("INTS", AttributeProto::INTS)
      .value("STRINGS", AttributeProto::STRINGS)
      .value("TENSORS", AttributeProto::TENSORS)
      .value("GRAPHS", AttributeProto::GRAPHS)
      .value("SPARSE_TENSOR", AttributeProto::SPARSE_TENSOR)
      .value("SPARSE_TENSORS", AttributeProto::SPARSE_TENSORS)
      .value("TYPE_PROTO", AttributeProto::TYPE_PROTO)
      .value("TYPE_PROTOS", AttributeProto::TYPE_PROTOS);

  nb::enum_<OpSchema::SupportType>(op_schema, "SupportType", nb::is_arithmetic())
      .value("COMMON", OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", OpSchema::SupportType::EXPERIMENTAL);

  nb::class_<OpSchema::Attribute>(op_schema, "Attribute")
      .def(
          "__init__",
          [](OpSchema::Attribute* self,
             std::string name,
             AttributeProto::AttributeType type,
             std::string description,
             bool required) {
            // Construct an attribute.
            // Use a lambda to swap the order of the arguments to match the Python API
            new (self) OpSchema::Attribute(std::move(name), std::move(description), type, required);
          },
          nb::arg("name"),
          nb::arg("type"),
          nb::arg("description") = "",
          nb::kw_only(),
          nb::arg("required") = true)
      .def(
          "__init__",
          [](OpSchema::Attribute* self, std::string name, const nb::object& default_value, std::string description) {
            // Construct an attribute with a default value.
            // Attributes with default values are not required
            auto bytes = nb::cast<nb::bytes>(default_value.attr("SerializeToString")());
            AttributeProto proto{};
            ParseProtoFromPyBytes(&proto, bytes);
            new (self) OpSchema::Attribute(std::move(name), std::move(description), std::move(proto));
          },
          nb::arg("name"),
          nb::arg("default_value"), // type: onnx.AttributeProto
          nb::arg("description") = "")
      .def_ro("name", &OpSchema::Attribute::name)
      .def_ro("description", &OpSchema::Attribute::description)
      .def_ro("type", &OpSchema::Attribute::type)
      .def_prop_ro(
          "_default_value",
          [](OpSchema::Attribute* attr) -> nb::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return nb::bytes(out.c_str(), out.size());
          })
      .def_ro("required", &OpSchema::Attribute::required);

  nb::class_<OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def(
          nb::init<std::string, std::vector<std::string>, std::string>(),
          nb::arg("type_param_str"),
          nb::arg("allowed_type_strs"),
          nb::arg("description") = "")
      .def_ro("type_param_str", &OpSchema::TypeConstraintParam::type_param_str)
      .def_ro("allowed_type_strs", &OpSchema::TypeConstraintParam::allowed_type_strs)
      .def_ro("description", &OpSchema::TypeConstraintParam::description);

  nb::class_<OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def(
          "__init__",
          [](OpSchema::FormalParameter* self,
             std::string name,
             std::string type_str,
             const std::string& description,
             OpSchema::FormalParameterOption param_option,
             bool is_homogeneous,
             int min_arity,
             OpSchema::DifferentiationCategory differentiation_category) {
            // Use a lambda to swap the order of the arguments to match the Python API
            new (self) OpSchema::FormalParameter(
                std::move(name),
                description,
                std::move(type_str),
                param_option,
                is_homogeneous,
                min_arity,
                differentiation_category);
          },
          nb::arg("name"),
          nb::arg("type_str"),
          nb::arg("description") = "",
          nb::kw_only(),
          nb::arg("param_option") = OpSchema::Single,
          nb::arg("is_homogeneous") = true,
          nb::arg("min_arity") = 1,
          nb::arg("differentiation_category") = OpSchema::DifferentiationCategory::Unknown)

      .def_prop_ro("name", &OpSchema::FormalParameter::GetName)
      .def_prop_ro("types", &OpSchema::FormalParameter::GetTypes)
      .def_prop_ro("type_str", &OpSchema::FormalParameter::GetTypeStr)
      .def_prop_ro("description", &OpSchema::FormalParameter::GetDescription)
      .def_prop_ro("option", &OpSchema::FormalParameter::GetOption)
      .def_prop_ro("is_homogeneous", &OpSchema::FormalParameter::GetIsHomogeneous)
      .def_prop_ro("min_arity", &OpSchema::FormalParameter::GetMinArity)
      .def_prop_ro("differentiation_category", &OpSchema::FormalParameter::GetDifferentiationCategory);

  op_schema
      .def(
          "__init__",
          [](OpSchema* self,
             std::string name,
             std::string domain,
             int since_version,
             const std::string& doc,
             std::vector<OpSchema::FormalParameter> inputs,
             std::vector<OpSchema::FormalParameter> outputs,
             std::vector<std::tuple<std::string, std::vector<std::string>, std::string>> type_constraints,
             std::vector<OpSchema::Attribute> attributes,
             OpSchema::NodeDeterminism node_determinism) {
            new (self) OpSchema();

            self->SetName(std::move(name)).SetDomain(std::move(domain)).SinceVersion(since_version).SetDoc(doc);
            self->SetNodeDeterminism(node_determinism);
            // Add inputs and outputs
            for (size_t i = 0; i < inputs.size(); ++i) {
              self->Input(static_cast<int>(i), std::move(inputs[i]));
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
              self->Output(static_cast<int>(i), std::move(outputs[i]));
            }
            // Add type constraints
            for (auto& type_constraint : type_constraints) {
              std::string type_str;
              std::vector<std::string> constraints;
              std::string description;
              tie(type_str, constraints, description) = std::move(type_constraint);
              self->TypeConstraint(std::move(type_str), std::move(constraints), std::move(description));
            }
            // Add attributes
            for (auto& attribute : attributes) {
              self->Attr(std::move(attribute));
            }

            self->Finalize();
          },
          nb::arg("name"),
          nb::arg("domain"),
          nb::arg("since_version"),
          nb::arg("doc") = "",
          nb::kw_only(),
          nb::arg("inputs") = std::vector<OpSchema::FormalParameter>{},
          nb::arg("outputs") = std::vector<OpSchema::FormalParameter>{},
          nb::arg("type_constraints") = std::vector<std::tuple<
              std::string /* type_str */,
              std::vector<std::string> /* constraints */,
              std::string /* description */>>{},
          nb::arg("attributes") = std::vector<OpSchema::Attribute>{},
          nb::arg("node_determinism") = OpSchema::NodeDeterminism::Unknown)
      .def_prop_rw("name", &OpSchema::Name, [](OpSchema& self, const std::string& name) { self.SetName(name); })
      .def_prop_rw(
          "domain", &OpSchema::domain, [](OpSchema& self, const std::string& domain) { self.SetDomain(domain); })
      .def_prop_rw("doc", &OpSchema::doc, [](OpSchema& self, const std::string& doc) { self.SetDoc(doc); })
      .def_prop_ro("file", &OpSchema::file)
      .def_prop_ro("line", &OpSchema::line)
      .def_prop_ro("support_level", &OpSchema::support_level)
      .def_prop_ro("since_version", &OpSchema::since_version)
      .def_prop_ro("deprecated", &OpSchema::deprecated)
      .def_prop_ro("function_opset_versions", &OpSchema::function_opset_versions)
      .def_prop_ro("context_dependent_function_opset_versions", &OpSchema::context_dependent_function_opset_versions)
      .def_prop_ro(
          "all_function_opset_versions",
          [](OpSchema* op) -> std::vector<int> {
            std::vector<int> all_function_opset_versions = op->function_opset_versions();
            std::vector<int> context_dependent_function_opset_versions =
                op->context_dependent_function_opset_versions();
            all_function_opset_versions.insert(
                all_function_opset_versions.end(),
                context_dependent_function_opset_versions.begin(),
                context_dependent_function_opset_versions.end());
            std::sort(all_function_opset_versions.begin(), all_function_opset_versions.end());
            all_function_opset_versions.erase(
                std::unique(all_function_opset_versions.begin(), all_function_opset_versions.end()),
                all_function_opset_versions.end());
            return all_function_opset_versions;
          })
      .def_prop_ro("min_input", &OpSchema::min_input)
      .def_prop_ro("max_input", &OpSchema::max_input)
      .def_prop_ro("min_output", &OpSchema::min_output)
      .def_prop_ro("max_output", &OpSchema::max_output)
      .def_prop_ro("attributes", &OpSchema::attributes)
      .def_prop_ro("inputs", &OpSchema::inputs)
      .def_prop_ro("outputs", &OpSchema::outputs)
      .def_prop_ro("has_type_and_shape_inference_function", &OpSchema::has_type_and_shape_inference_function)
      .def_prop_ro("has_data_propagation_function", &OpSchema::has_data_propagation_function)
      .def_prop_ro("type_constraints", &OpSchema::typeConstraintParams)
      .def_prop_ro("node_determinism", &OpSchema::GetNodeDeterminism)
      .def_static("is_infinite", [](int v) { return v == std::numeric_limits<int>::max(); })
      .def(
          "_infer_node_outputs",
          CallNodeInferenceFunction,
          nb::arg("nodeBytes"),
          nb::arg("valueTypesByNameBytes"),
          nb::arg("inputDataByNameBytes") = std::unordered_map<std::string, nb::bytes>{},
          nb::arg("inputSparseDataByNameBytes") = std::unordered_map<std::string, nb::bytes>{},
          nb::arg("opsetImports") = std::unordered_map<std::string, int>{},
          nb::arg("irVersion") = int(IR_VERSION))
      .def_prop_ro("has_function", &OpSchema::HasFunction)
      .def_prop_ro(
          "_function_body",
          [](OpSchema* op) -> nb::bytes {
            std::string bytes = "";
            if (op->HasFunction())
              op->GetFunction()->SerializeToString(&bytes);
            return nb::bytes(bytes.c_str(), bytes.size());
          })
      .def(
          "get_function_with_opset_version",
          [](OpSchema* op, int opset_version) -> nb::bytes {
            std::string bytes = "";
            const FunctionProto* function_proto = op->GetFunction(opset_version);
            if (function_proto) {
              function_proto->SerializeToString(&bytes);
            }
            return nb::bytes(bytes.c_str(), bytes.size());
          })
      .def_prop_ro("has_context_dependent_function", &OpSchema::HasContextDependentFunction)
      .def(
          "get_context_dependent_function",
          [](OpSchema* op, const nb::bytes& bytes, const std::vector<nb::bytes>& input_types_bytes) -> nb::bytes {
            NodeProto proto{};
            ParseProtoFromPyBytes(&proto, bytes);
            std::string func_bytes = "";
            if (op->HasContextDependentFunction()) {
              std::vector<TypeProto> input_types;
              input_types.reserve(input_types_bytes.size());
              for (auto& type_bytes : input_types_bytes) {
                TypeProto type_proto{};
                ParseProtoFromPyBytes(&type_proto, type_bytes);
                input_types.push_back(type_proto);
              }
              FunctionBodyBuildContextImpl ctx(proto, input_types);
              FunctionProto func_proto;
              op->BuildContextDependentFunction(ctx, func_proto);
              func_proto.SerializeToString(&func_bytes);
            }
            return nb::bytes(func_bytes.c_str(), func_bytes.size());
          })
      .def(
          "get_context_dependent_function_with_opset_version",
          [](OpSchema* op, int opset_version, const nb::bytes& bytes, const std::vector<nb::bytes>& input_types_bytes)
              -> nb::bytes {
            NodeProto proto{};
            ParseProtoFromPyBytes(&proto, bytes);
            std::string func_bytes = "";
            if (op->HasContextDependentFunctionWithOpsetVersion(opset_version)) {
              std::vector<TypeProto> input_types;
              input_types.reserve(input_types_bytes.size());
              for (auto& type_bytes : input_types_bytes) {
                TypeProto type_proto{};
                ParseProtoFromPyBytes(&type_proto, type_bytes);
                input_types.push_back(type_proto);
              }
              FunctionBodyBuildContextImpl ctx(proto, input_types);
              FunctionProto func_proto;
              op->BuildContextDependentFunction(ctx, func_proto, opset_version);
              func_proto.SerializeToString(&func_bytes);
            }
            return nb::bytes(func_bytes.c_str(), func_bytes.size());
          })
      .def(
          "set_type_and_shape_inference_function",
          [](OpSchema& op, const std::function<void(InferenceContext*)>& func) -> OpSchema& {
            auto wrapper = [=](InferenceContext& ctx) { func(&ctx); };
            return op.TypeAndShapeInferenceFunction(wrapper);
          },
          nb::rv_policy::reference_internal)
      .def("get_type_and_shape_inference_function", &OpSchema::GetTypeAndShapeInferenceFunction);

  defs.def(
          "has_schema",
          [](const std::string& op_type, const std::string& domain) -> bool {
            return OpSchemaRegistry::Schema(op_type, domain) != nullptr;
          },
          "op_type"_a,
          "domain"_a = ONNX_DOMAIN)
      .def(
          "has_schema",
          [](const std::string& op_type, int max_inclusive_version, const std::string& domain) -> bool {
            return OpSchemaRegistry::Schema(op_type, max_inclusive_version, domain) != nullptr;
          },
          "op_type"_a,
          "max_inclusive_version"_a,
          "domain"_a = ONNX_DOMAIN)
      .def(
          "schema_version_map",
          []() -> std::unordered_map<std::string, std::pair<int, int>> {
            return OpSchemaRegistry::DomainToVersionRange::Instance().Map();
          })
      .def(
          "get_schema",
          [](const std::string& op_type, const int max_inclusive_version, const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, max_inclusive_version, domain);
            if (!schema) {
              fail_schema(
                  "No schema registered for '" + op_type + "' version '" + std::to_string(max_inclusive_version) +
                  "' and domain '" + domain + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "max_inclusive_version"_a,
          "domain"_a = ONNX_DOMAIN,
          "Return the schema of the operator *op_type* and for a specific version.")
      .def(
          "get_schema",
          [](const std::string& op_type, const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, domain);
            if (!schema) {
              fail_schema("No schema registered for '" + op_type + "' and domain '" + domain + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "domain"_a = ONNX_DOMAIN,
          "Return the schema of the operator *op_type* and for a specific version.")
      .def(
          "get_all_schemas",
          []() -> const std::vector<OpSchema> { return OpSchemaRegistry::get_all_schemas(); },
          "Return the schema of all existing operators for the latest version.")
      .def(
          "get_all_schemas_with_history",
          []() -> const std::vector<OpSchema> { return OpSchemaRegistry::get_all_schemas_with_history(); },
          "Return the schema of all existing operators and all versions.")
      .def(
          "set_domain_to_version",
          [](const std::string& domain, int min_version, int max_version, int last_release_version) {
            auto& obj = OpSchemaRegistry::DomainToVersionRange::Instance();
            if (obj.Map().count(domain) == 0) {
              obj.AddDomainToVersion(domain, min_version, max_version, last_release_version);
            } else {
              obj.UpdateDomainToVersion(domain, min_version, max_version, last_release_version);
            }
          },
          "domain"_a,
          "min_version"_a,
          "max_version"_a,
          "last_release_version"_a = -1,
          "Set the version range and last release version of the specified domain.")
      .def(
          "register_schema",
          [](OpSchema schema) { RegisterSchema(std::move(schema), 0, true, true); },
          "schema"_a,
          "Register a user provided OpSchema.")
      .def(
          "deregister_schema",
          &DeregisterSchema,
          "op_type"_a,
          "version"_a,
          "domain"_a,
          "Deregister the specified OpSchema.");

  // Submodule `checker`
  auto checker = onnx_cpp2py_export.def_submodule("checker");
  checker.doc() = "Checker submodule";

  nb::class_<checker::CheckerContext> checker_context(checker, "CheckerContext");
  checker_context.def(nb::init<>())
      .def_prop_rw("ir_version", &checker::CheckerContext::get_ir_version, &checker::CheckerContext::set_ir_version)
      .def_prop_rw(
          "opset_imports", &checker::CheckerContext::get_opset_imports, &checker::CheckerContext::set_opset_imports);

  nb::class_<checker::LexicalScopeContext> lexical_scope_context(checker, "LexicalScopeContext");
  lexical_scope_context.def(nb::init<>());

  nb::exception<checker::ValidationError>(checker, "ValidationError");

  checker.def("check_value_info", [](const nb::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    ValueInfoProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_value_info(proto, ctx);
  });

  checker.def("check_tensor", [](const nb::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    TensorProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_tensor(proto, ctx);
  });

  checker.def("check_sparse_tensor", [](const nb::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    SparseTensorProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_sparse_tensor(proto, ctx);
  });

  checker.def(
      "check_attribute",
      [](const nb::bytes& bytes,
         const checker::CheckerContext& ctx,
         const checker::LexicalScopeContext& lex_ctx) -> void {
        AttributeProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_attribute(proto, ctx, lex_ctx);
      });

  checker.def(
      "check_node",
      [](const nb::bytes& bytes,
         const checker::CheckerContext& ctx,
         const checker::LexicalScopeContext& lex_ctx) -> void {
        NodeProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_node(proto, ctx, lex_ctx);
      });

  checker.def(
      "check_function",
      [](const nb::bytes& bytes,
         const checker::CheckerContext& ctx,
         const checker::LexicalScopeContext& lex_ctx) -> void {
        FunctionProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_function(proto, ctx, lex_ctx);
      });

  checker.def(
      "check_graph",
      [](const nb::bytes& bytes,
         const checker::CheckerContext& ctx,
         const checker::LexicalScopeContext& lex_ctx) -> void {
        GraphProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_graph(proto, ctx, lex_ctx);
      });

  checker.def(
      "check_model",
      [](const nb::bytes& bytes, bool full_check, bool skip_opset_compatibility_check, bool check_custom_domain)
          -> void {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_model(proto, full_check, skip_opset_compatibility_check, check_custom_domain);
      },
      "bytes"_a,
      "full_check"_a = false,
      "skip_opset_compatibility_check"_a = false,
      "check_custom_domain"_a = false);

  checker.def(
      "check_model_path",
      (void (*)(
          const std::string& path,
          bool full_check,
          bool skip_opset_compatibility_check,
          bool check_custom_domain))&checker::check_model,
      "path"_a,
      "full_check"_a = false,
      "skip_opset_compatibility_check"_a = false,
      "check_custom_domain"_a = false);

  checker.def("_resolve_external_data_location", &checker::resolve_external_data_location);

  // Submodule `version_converter`
  auto version_converter = onnx_cpp2py_export.def_submodule("version_converter");
  version_converter.doc() = "VersionConverter submodule";
  nb::exception<ConvertError>(version_converter, "ConvertError");

  version_converter.def("convert_version", [](const nb::bytes& bytes, int target) {
    ModelProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    shape_inference::InferShapes(proto);
    auto result = version_conversion::ConvertVersion(proto, target);
    std::string out;
    result.SerializeToString(&out);
    return nb::bytes(out.c_str(), out.size());
  });

  // Submodule `inliner`
  auto inliner = onnx_cpp2py_export.def_submodule("inliner");
  inliner.doc() = "Inliner submodule";

  inliner.def("inline_local_functions", [](const nb::bytes& bytes, bool convert_version) {
    ModelProto model{};
    ParseProtoFromPyBytes(&model, bytes);
    inliner::InlineLocalFunctions(model, convert_version);
    std::string out;
    model.SerializeToString(&out);
    return nb::bytes(out.c_str(), out.size());
  });

  // inline_selected_functions: Inlines all functions specified in function_ids, unless
  // exclude is true, in which case it inlines all functions except those specified in
  // function_ids.
  inliner.def(
      "inline_selected_functions",
      [](const nb::bytes& bytes, std::vector<std::pair<std::string, std::string>> function_ids, bool exclude) {
        ModelProto model{};
        ParseProtoFromPyBytes(&model, bytes);
        auto function_id_set = inliner::FunctionIdSet::Create(std::move(function_ids), exclude);
        inliner::InlineSelectedLocalFunctions(model, *function_id_set);
        std::string out;
        model.SerializeToString(&out);
        return nb::bytes(out.c_str(), out.size());
      });

  inliner.def(
      "inline_selected_functions2",
      [](const nb::bytes& bytes, std::vector<std::pair<std::string, std::string>> function_ids, bool exclude) {
        ModelProto model{};
        ParseProtoFromPyBytes(&model, bytes);
        auto function_id_set = inliner::FunctionIdSet::Create(std::move(function_ids), exclude);
        inliner::InlineSelectedFunctions(model, *function_id_set, nullptr);
        std::string out;
        model.SerializeToString(&out);
        return nb::bytes(out.c_str(), out.size());
      });

  // Submodule `shape_inference`
  auto shape_inference = onnx_cpp2py_export.def_submodule("shape_inference");
  shape_inference.doc() = "Shape Inference submodule";
  nb::exception<InferenceError>(shape_inference, "InferenceError");

  nb::class_<InferenceContext> inference_context(shape_inference, "InferenceContext", "Inference context");

  inference_context.def("get_attribute", [](InferenceContext& self, const std::string& name) -> nb::object {
    const auto* attr = self.getAttribute(name);
    if (attr == nullptr) {
      return nb::none();
    }
    return nb::cast(*attr);
  });
  inference_context.def("get_num_inputs", &InferenceContext::getNumInputs);
  inference_context.def("get_input_type", [](InferenceContext& self, size_t idx) -> nb::object {
    const auto* type = self.getInputType(idx);
    if (type == nullptr) {
      return nb::none();
    }
    return nb::cast(*type);
  });
  inference_context.def("has_input", &InferenceContext::hasInput);
  inference_context.def("get_input_data", [](InferenceContext& self, size_t idx) -> nb::object {
    const auto* tensor = self.getInputData(idx);
    if (tensor == nullptr) {
      return nb::none();
    }
    return nb::cast(*tensor);
  });
  inference_context.def("get_num_outputs", &InferenceContext::getNumOutputs);
  inference_context.def("get_output_type", [](InferenceContext& self, size_t idx) -> nb::object {
    const auto* type = self.getOutputType(idx);
    if (type == nullptr) {
      return nb::none();
    }
    return nb::cast(*type);
  });
  inference_context.def("set_output_type", [](InferenceContext& self, size_t idx, const TypeProto& src) {
    auto* dst = self.getOutputType(idx);
    if (dst == nullptr) {
      return false;
    }
    dst->CopyFrom(src);
    return true;
  });
  inference_context.def("has_output", &InferenceContext::hasOutput);
  inference_context.def(
      "get_graph_attribute_inferencer",
      &InferenceContext::getGraphAttributeInferencer,
      nb::rv_policy::reference_internal);
  inference_context.def("get_input_sparse_data", [](InferenceContext& self, size_t idx) -> nb::object {
    const auto* sparse = self.getInputSparseData(idx);
    if (sparse == nullptr) {
      return nb::none();
    }
    return nb::cast(*sparse);
  });
  inference_context.def("get_symbolic_input", [](InferenceContext& self, size_t idx) -> nb::object {
    const auto* shape = self.getSymbolicInput(idx);
    if (shape == nullptr) {
      return nb::none();
    }
    return nb::cast(*shape);
  });
  inference_context.def("get_display_name", &InferenceContext::getDisplayName);

  nb::class_<GraphInferencer> graph_inferencer(shape_inference, "GraphInferencer", "Graph Inferencer");
  graph_inferencer.def(
      "do_inferencing",
      [](GraphInferencer& self,
         const std::vector<nb::object>& inputTypesObj,
         const std::vector<nb::object>& inputDataObj) {
        auto inputTypesTuple = ConvertPyObjToPtr<ONNX_NAMESPACE::TypeProto>(inputTypesObj);
        auto inputDataTuple = ConvertPyObjToPtr<ONNX_NAMESPACE::TensorProto>(inputDataObj);
        auto ret = self.doInferencing(std::get<1>(inputTypesTuple), std::get<1>(inputDataTuple));
        std::vector<nb::object> ret_obj(ret.size());
        for (size_t i = 0; i < ret.size(); ++i) {
          ret_obj[i] = nb::cast(ret[i]);
        }
        return ret_obj;
      });

  shape_inference.def(
      "infer_shapes",
      [](const nb::bytes& bytes, bool check_type, bool strict_mode, bool data_prop) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        ShapeInferenceOptions options{check_type, strict_mode ? 1 : 0, data_prop};
        shape_inference::InferShapes(proto, OpSchemaRegistry::Instance(), options);
        std::string out;
        proto.SerializeToString(&out);
        return nb::bytes(out.c_str(), out.size());
      },
      "bytes"_a,
      "check_type"_a = false,
      "strict_mode"_a = false,
      "data_prop"_a = false);

  shape_inference.def(
      "infer_shapes_path",
      [](const std::string& model_path,
         const std::string& output_path,
         bool check_type,
         bool strict_mode,
         bool data_prop) -> void {
        ShapeInferenceOptions options{check_type, strict_mode ? 1 : 0, data_prop};
        shape_inference::InferShapes(model_path, output_path, OpSchemaRegistry::Instance(), options);
      });

  shape_inference.def(
      "infer_function_output_types",
      [](const nb::bytes& function_proto_bytes,
         const std::vector<nb::bytes>& input_types_bytes,
         const std::vector<nb::bytes>& attributes_bytes) -> std::vector<nb::bytes> {
        FunctionProto proto{};
        ParseProtoFromPyBytes(&proto, function_proto_bytes);

        std::vector<TypeProto> input_types;
        input_types.reserve(input_types_bytes.size());
        for (const nb::bytes& bytes : input_types_bytes) {
          TypeProto type;
          ParseProtoFromPyBytes(&type, bytes);
          input_types.push_back(type);
        }

        std::vector<AttributeProto> attributes;
        attributes.reserve(attributes_bytes.size());
        for (const nb::bytes& bytes : attributes_bytes) {
          AttributeProto attr;
          ParseProtoFromPyBytes(&attr, bytes);
          attributes.push_back(attr);
        }

        std::vector<TypeProto> output_types = shape_inference::InferFunctionOutputTypes(proto, input_types, attributes);
        std::vector<nb::bytes> result;
        result.reserve(output_types.size());
        for (auto& type_proto : output_types) {
          std::string out;
          type_proto.SerializeToString(&out);
          result.emplace_back(out.c_str(), out.size());
        }
        return result;
      });

  // Submodule `parser`
  auto parser = onnx_cpp2py_export.def_submodule("parser");
  parser.doc() = "Parser submodule";

  parser.def("parse_model", Parse<ModelProto>);
  parser.def("parse_graph", Parse<GraphProto>);
  parser.def("parse_function", Parse<FunctionProto>);
  parser.def("parse_node", Parse<NodeProto>);

  // Submodule `printer`
  auto printer = onnx_cpp2py_export.def_submodule("printer");
  printer.doc() = "Printer submodule";

  printer.def("model_to_text", ProtoBytesToText<ModelProto>);
  printer.def("function_to_text", ProtoBytesToText<FunctionProto>);
  printer.def("graph_to_text", ProtoBytesToText<GraphProto>);
}

} // namespace ONNX_NAMESPACE
