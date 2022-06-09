/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <climits>
#include <limits>
#include <tuple>
#include <unordered_map>

#include "onnx/checker.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/py_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE {
namespace py = pybind11;
using namespace pybind11::literals;

template <typename ProtoType>
static std::tuple<bool, py::bytes, py::bytes> Parse(const char* cstr) {
  ProtoType proto{};
  OnnxParser parser(cstr);
  auto status = parser.Parse(proto);
  std::string out;
  proto.SerializeToString(&out);
  return std::make_tuple(status.IsOK(), py::bytes(status.ErrorMessage()), py::bytes(out));
}

PYBIND11_MODULE(onnx_cpp2py_export, onnx_cpp2py_export) {
  onnx_cpp2py_export.doc() = "Python interface to onnx";

  onnx_cpp2py_export.attr("ONNX_ML") = py::bool_(
#ifdef ONNX_ML
      true
#else // ONNX_ML
      false
#endif // ONNX_ML
  );

  // Submodule `schema`
  auto defs = onnx_cpp2py_export.def_submodule("defs");
  defs.doc() = "Schema submodule";
  py::register_exception<SchemaError>(defs, "SchemaError");

  py::class_<OpSchema> op_schema(defs, "OpSchema");
  op_schema.def_property_readonly("file", &OpSchema::file)
      .def_property_readonly("line", &OpSchema::line)
      .def_property_readonly("support_level", &OpSchema::support_level)
      .def_property_readonly("doc", &OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("since_version", &OpSchema::since_version)
      .def_property_readonly("deprecated", &OpSchema::deprecated)
      .def_property_readonly("domain", &OpSchema::domain)
      .def_property_readonly("name", &OpSchema::Name)
      .def_property_readonly("min_input", &OpSchema::min_input)
      .def_property_readonly("max_input", &OpSchema::max_input)
      .def_property_readonly("min_output", &OpSchema::min_output)
      .def_property_readonly("max_output", &OpSchema::max_output)
      .def_property_readonly("attributes", &OpSchema::attributes)
      .def_property_readonly("inputs", &OpSchema::inputs)
      .def_property_readonly("outputs", &OpSchema::outputs)
      .def_property_readonly("has_type_and_shape_inference_function", &OpSchema::has_type_and_shape_inference_function)
      .def_property_readonly("has_data_propagation_function", &OpSchema::has_data_propagation_function)
      .def_property_readonly("type_constraints", &OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) { return v == std::numeric_limits<int>::max(); })
      .def_property_readonly("has_function", &OpSchema::HasFunction)
      .def_property_readonly(
          "_function_body",
          [](OpSchema* op) -> py::bytes {
            std::string bytes = "";
            if (op->HasFunction())
              op->GetFunction()->SerializeToString(&bytes);
            return py::bytes(bytes);
          })
      .def_property_readonly("has_context_dependent_function", &OpSchema::HasContextDependentFunction)
      .def(
          "get_context_dependent_function",
          [](OpSchema* op, const py::bytes& bytes, const std::vector<py::bytes>& input_types_bytes) -> py::bytes {
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
            return py::bytes(func_bytes);
          });
  ;

  py::class_<OpSchema::Attribute>(op_schema, "Attribute")
      .def_readonly("name", &OpSchema::Attribute::name)
      .def_readonly("description", &OpSchema::Attribute::description)
      .def_readonly("type", &OpSchema::Attribute::type)
      .def_property_readonly(
          "_default_value",
          [](OpSchema::Attribute* attr) -> py::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return out;
          })
      .def_readonly("required", &OpSchema::Attribute::required);

  py::class_<OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def_readonly("type_param_str", &OpSchema::TypeConstraintParam::type_param_str)
      .def_readonly("description", &OpSchema::TypeConstraintParam::description)
      .def_readonly("allowed_type_strs", &OpSchema::TypeConstraintParam::allowed_type_strs);

  py::enum_<OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption")
      .value("Single", OpSchema::Single)
      .value("Optional", OpSchema::Optional)
      .value("Variadic", OpSchema::Variadic);

  py::enum_<OpSchema::DifferentiationCategory>(op_schema, "DifferentiationCategory")
      .value("Unknown", OpSchema::Unknown)
      .value("Differentiable", OpSchema::Differentiable)
      .value("NonDifferentiable", OpSchema::NonDifferentiable);

  py::class_<OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def_property_readonly("name", &OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("typeStr", &OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly("description", &OpSchema::FormalParameter::GetDescription)
      .def_property_readonly("option", &OpSchema::FormalParameter::GetOption)
      .def_property_readonly("isHomogeneous", &OpSchema::FormalParameter::GetIsHomogeneous)
      .def_property_readonly("differentiationCategory", &OpSchema::FormalParameter::GetDifferentiationCategory);

  py::enum_<AttributeProto::AttributeType>(op_schema, "AttrType")
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

  py::enum_<OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", OpSchema::SupportType::EXPERIMENTAL);

  defs.def(
      "has_schema",
      [](const std::string& op_type, const std::string& domain) -> bool {
        return OpSchemaRegistry::Schema(op_type, domain) != nullptr;
      },
      "op_type"_a,
      "domain"_a = ONNX_DOMAIN);
  defs.def("schema_version_map", []() -> std::unordered_map<std::string, std::pair<int, int>> {
    return OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  });
  defs.def(
          "get_schema",
          [](const std::string& op_type, const int max_inclusive_version, const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, max_inclusive_version, domain);
            if (!schema) {
              fail_schema("No schema registered for '" + op_type + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "max_inclusive_version"_a,
          "domain"_a = ONNX_DOMAIN)
      .def(
          "get_schema",
          [](const std::string& op_type, const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, domain);
            if (!schema) {
              fail_schema("No schema registered for '" + op_type + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "domain"_a = ONNX_DOMAIN);

  defs.def("get_all_schemas", []() -> const std::vector<OpSchema> { return OpSchemaRegistry::get_all_schemas(); });

  defs.def("get_all_schemas_with_history", []() -> const std::vector<OpSchema> {
    return OpSchemaRegistry::get_all_schemas_with_history();
  });

  // Submodule `checker`
  auto checker = onnx_cpp2py_export.def_submodule("checker");
  checker.doc() = "Checker submodule";

  py::class_<checker::CheckerContext> checker_context(checker, "CheckerContext");
  checker_context.def(py::init<>())
      .def_property("ir_version", &checker::CheckerContext::get_ir_version, &checker::CheckerContext::set_ir_version)
      .def_property(
          "opset_imports", &checker::CheckerContext::get_opset_imports, &checker::CheckerContext::set_opset_imports);

  py::register_exception<checker::ValidationError>(checker, "ValidationError");

  checker.def("check_value_info", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    ValueInfoProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_value_info(proto, ctx);
  });

  checker.def("check_tensor", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    TensorProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_tensor(proto, ctx);
  });

  checker.def("check_sparse_tensor", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    SparseTensorProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_sparse_tensor(proto, ctx);
  });

  checker.def("check_attribute", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    AttributeProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_attribute(proto, ctx, checker::LexicalScopeContext());
  });

  checker.def("check_node", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    NodeProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::LexicalScopeContext lex_ctx;
    checker::check_node(proto, ctx, lex_ctx);
  });

  checker.def("check_graph", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    GraphProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::LexicalScopeContext lex_ctx;
    checker::check_graph(proto, ctx, lex_ctx);
  });

  checker.def("check_model", [](const py::bytes& bytes) -> void {
    ModelProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_model(proto);
  });

  checker.def("check_model_path", (void (*)(const std::string&)) & checker::check_model);

  // Submodule `version_converter`
  auto version_converter = onnx_cpp2py_export.def_submodule("version_converter");
  version_converter.doc() = "VersionConverter submodule";
  py::register_exception<ConvertError>(version_converter, "ConvertError");

  version_converter.def("convert_version", [](const py::bytes& bytes, py::int_ target) {
    ModelProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    shape_inference::InferShapes(proto);
    auto result = version_conversion::ConvertVersion(proto, target);
    std::string out;
    result.SerializeToString(&out);
    return py::bytes(out);
  });

  // Submodule `shape_inference`
  auto shape_inference = onnx_cpp2py_export.def_submodule("shape_inference");
  shape_inference.doc() = "Shape Inference submodule";
  py::register_exception<InferenceError>(shape_inference, "InferenceError");

  shape_inference.def(
      "infer_shapes",
      [](const py::bytes& bytes, bool check_type, bool strict_mode, bool data_prop) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        ShapeInferenceOptions options{check_type, strict_mode == true ? 1 : 0, data_prop};
        shape_inference::InferShapes(proto, OpSchemaRegistry::Instance(), options);
        std::string out;
        proto.SerializeToString(&out);
        return py::bytes(out);
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
        ShapeInferenceOptions options{check_type, strict_mode == true ? 1 : 0, data_prop};
        shape_inference::InferShapes(model_path, output_path, OpSchemaRegistry::Instance(), options);
      });

  // Submodule `parser`
  auto parser = onnx_cpp2py_export.def_submodule("parser");
  parser.doc() = "Parser submodule";

  parser.def("parse_model", Parse<ModelProto>);
  parser.def("parse_graph", Parse<GraphProto>);
}

} // namespace ONNX_NAMESPACE
