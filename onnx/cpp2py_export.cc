#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <climits>
#include <limits>
#include <unordered_map>

#include "onnx/checker.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/optimizer/optimize.h"
#include "onnx/py_utils.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(onnx_cpp2py_export, onnx_cpp2py_export) {
  onnx_cpp2py_export.doc() = "Python interface to onnx";

  // Submodule `schema`
  auto defs = onnx_cpp2py_export.def_submodule("defs");
  defs.doc() = "Schema submodule";

  py::class_<OpSchema> op_schema(defs, "OpSchema");
  op_schema.def_property_readonly("file", &OpSchema::file)
      .def_property_readonly("line", &OpSchema::line)
      .def_property_readonly("support_level", &OpSchema::support_level)
      .def_property_readonly(
          "doc", &OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("since_version", &OpSchema::since_version)
      .def_property_readonly("domain", &OpSchema::domain)
      .def_property_readonly("name", &OpSchema::Name)
      .def_property_readonly("min_input", &OpSchema::min_input)
      .def_property_readonly("max_input", &OpSchema::max_input)
      .def_property_readonly("min_output", &OpSchema::min_output)
      .def_property_readonly("max_output", &OpSchema::max_output)
      .def_property_readonly("attributes", &OpSchema::attributes)
      .def_property_readonly("inputs", &OpSchema::inputs)
      .def_property_readonly("outputs", &OpSchema::outputs)
      .def_property_readonly(
          "has_type_and_shape_inference_function",
          &OpSchema::has_type_and_shape_inference_function)
      .def_property_readonly(
          "type_constraints", &OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) {
        return v == std::numeric_limits<int>::max();
      });

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
      .def_readonly(
          "type_param_str", &OpSchema::TypeConstraintParam::type_param_str)
      .def_readonly("description", &OpSchema::TypeConstraintParam::description)
      .def_readonly(
          "allowed_type_strs",
          &OpSchema::TypeConstraintParam::allowed_type_strs);

  py::enum_<OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption")
      .value("Single", OpSchema::Single)
      .value("Optional", OpSchema::Optional)
      .value("Variadic", OpSchema::Variadic);

  py::class_<OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def_property_readonly("name", &OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("typeStr", &OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly(
          "description", &OpSchema::FormalParameter::GetDescription)
      .def_property_readonly("option", &OpSchema::FormalParameter::GetOption);

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
      .value("GRAPHS", AttributeProto::GRAPHS);

  py::enum_<OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", OpSchema::SupportType::EXPERIMENTAL);

  py::class_<FunctionProto> function_proto(defs, "FunctionProto");
  function_proto.def_property_readonly("name", &FunctionProto::name)
      .def_property_readonly("doc_string", &FunctionProto::doc_string)
      .def_property_readonly("since_version", &FunctionProto::since_version)
      .def_property_readonly(
          "inputs",
          [](FunctionProto* fp) -> std::vector<std::string> {
            std::vector<std::string> _stl_vec;
            for (auto ptr = fp->input().begin(); ptr != fp->input().end();
                 ++ptr) {
              _stl_vec.emplace_back(*ptr);
            }
            return _stl_vec;
          })
      .def_property_readonly(
          "outputs",
          [](FunctionProto* fp) -> std::vector<std::string> {
            std::vector<std::string> _stl_vec;
            for (auto ptr = fp->output().begin(); ptr != fp->output().end();
                 ++ptr) {
              _stl_vec.emplace_back(*ptr);
            }
            return _stl_vec;
          })
      .def_property_readonly(
          "attribute",
          [](FunctionProto* fp) -> std::vector<std::string> {
            std::vector<std::string> _stl_vec;
            for (auto ptr = fp->attribute().begin();
                 ptr != fp->attribute().end();
                 ++ptr) {
              _stl_vec.emplace_back(*ptr);
            }
            return _stl_vec;
          })
      .def_property_readonly(
          "nodes", [](FunctionProto* fp) -> std::vector<NodeProto> {
            std::vector<NodeProto> _stl_vec;
            for (auto ptr = fp->node().begin(); ptr != fp->node().end();
                 ++ptr) {
              _stl_vec.emplace_back(*ptr);
            }
            return _stl_vec;
          });

  py::class_<NodeProto> node_proto(function_proto, "NodeProto");
  node_proto.def_property_readonly("name", &NodeProto::name)
      .def_property_readonly("doc_string", &NodeProto::doc_string)
      .def_property_readonly("domain", &NodeProto::domain)
      .def_property_readonly("op_type", &NodeProto::op_type)
      .def_property_readonly(
          "inputs",
          [](NodeProto* np) -> std::vector<std::string> {
            std::vector<std::string> _stl_vec;
            for (auto ptr = np->input().begin(); ptr != np->input().end();
                 ++ptr) {
              _stl_vec.emplace_back(*ptr);
            }
            return _stl_vec;
          })
      .def_property_readonly(
          "outputs", [](NodeProto* np) -> std::vector<std::string> {
            std::vector<std::string> _stl_vec;
            for (auto ptr = np->output().begin(); ptr != np->output().end();
                 ++ptr) {
              _stl_vec.emplace_back(*ptr);
            }
            return _stl_vec;
          });

  defs.def(
      "has_schema",
      [](const std::string& op_type, const std::string& domain) -> bool {
        return OpSchemaRegistry::Schema(op_type, domain) != nullptr;
      },
      "op_type"_a,
      "domain"_a = ONNX_DOMAIN);
  defs.def(
      "schema_version_map",
      []() -> std::unordered_map<std::string, std::pair<int, int>> {
        return OpSchemaRegistry::DomainToVersionRange::Instance().Map();
      });
  defs.def(
          "get_schema",
          [](const std::string& op_type,
             const int max_inclusive_version,
             const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(
                op_type, max_inclusive_version, domain);
            if (!schema) {
              throw std::runtime_error(
                  "No schema registered for '" + op_type + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "max_inclusive_version"_a,
          "domain"_a = ONNX_DOMAIN)
      .def(
          "get_schema",
          [](const std::string& op_type,
             const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, domain);
            if (!schema) {
              throw std::runtime_error(
                  "No schema registered for '" + op_type + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "domain"_a = ONNX_DOMAIN);

  defs.def("get_all_schemas", []() -> const std::vector<OpSchema> {
    return OpSchemaRegistry::get_all_schemas();
  });

  defs.def("get_all_schemas_with_history", []() -> const std::vector<OpSchema> {
    return OpSchemaRegistry::get_all_schemas_with_history();
  });

  defs.def(
      "get_all_functions",
      [](const std::string& domain)
          -> std::unordered_map<std::string, FunctionProto> {
        std::multimap<std::string, std::unique_ptr<FunctionProto>> temp_ptr_map;
        std::unordered_map<std::string, FunctionProto> temp_map;
        FunctionBuilderRegistry& function_registry =
            FunctionBuilderRegistry::OnnxInstance();
        Common::Status status =
            function_registry.GetFunctions(domain, &temp_ptr_map);
        // Pybind not support stl with smart pointers well
        for (auto iter = temp_ptr_map.begin(); iter != temp_ptr_map.end();
             ++iter)
          temp_map.insert(
              std::unordered_map<std::string, FunctionProto>::value_type(
                  iter->first, *(iter->second)));
        return temp_map;
      });

  // Submodule `checker`
  auto checker = onnx_cpp2py_export.def_submodule("checker");
  checker.doc() = "Checker submodule";

  py::class_<checker::CheckerContext> checker_context(
      checker, "CheckerContext");
  checker_context.def(py::init<>())
      .def_property(
          "ir_version",
          &checker::CheckerContext::get_ir_version,
          &checker::CheckerContext::set_ir_version)
      .def_property(
          "opset_imports",
          &checker::CheckerContext::get_opset_imports,
          &checker::CheckerContext::set_opset_imports);

  py::register_exception<checker::ValidationError>(checker, "ValidationError");

  checker.def(
      "check_value_info",
      [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
        ValueInfoProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_value_info(proto, ctx);
      });

  checker.def(
      "check_tensor",
      [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
        TensorProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_tensor(proto, ctx);
      });

  checker.def(
      "check_attribute",
      [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
        AttributeProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_attribute(proto, ctx, checker::LexicalScopeContext());
      });

  checker.def(
      "check_node",
      [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
        NodeProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::LexicalScopeContext lex_ctx;
        checker::check_node(proto, ctx, lex_ctx);
      });

  checker.def(
      "check_graph",
      [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
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

  // Submodule `optimizer`
  auto optimizer = onnx_cpp2py_export.def_submodule("optimizer");
  optimizer.doc() = "Optimizer submodule";

  optimizer.def(
      "optimize",
      [](const py::bytes& bytes, const std::vector<std::string>& names) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        auto const result = optimization::Optimize(std::move(proto), names);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out);
      });

  // Submodule `shape_inference`
  auto shape_inference = onnx_cpp2py_export.def_submodule("shape_inference");
  shape_inference.doc() = "Shape Inference submodule";

  shape_inference.def("infer_shapes", [](const py::bytes& bytes) {
    ModelProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    shape_inference::InferShapes(proto);
    std::string out;
    proto.SerializeToString(&out);
    return py::bytes(out);
  });
}

} // namespace ONNX_NAMESPACE
