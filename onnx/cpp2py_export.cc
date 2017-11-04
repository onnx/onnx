#include <pybind11/pybind11.h>
#include <climits>
#include <limits>
#include <pybind11/stl.h>
#include <unordered_map>

#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"

namespace onnx {

namespace py = pybind11;

PYBIND11_MODULE(onnx_cpp2py_export, onnx_cpp2py_export) {
  onnx_cpp2py_export.doc() = "Python interface to onnx";

  // Submodule `schema`
  auto defs = onnx_cpp2py_export.def_submodule("defs");
  defs.doc() = "Schema submodule";

  py::class_<OpSchema> op_schema(defs, "OpSchema");
  op_schema
    .def_property_readonly("file", &OpSchema::file)
    .def_property_readonly("line", &OpSchema::line)
    .def_property_readonly("support_level", &OpSchema::support_level)
    .def_property_readonly(
      "doc", &OpSchema::doc, py::return_value_policy::reference)
    .def_property_readonly("min_input", &OpSchema::min_input)
    .def_property_readonly("max_input", &OpSchema::max_input)
    .def_property_readonly("min_output", &OpSchema::min_output)
    .def_property_readonly("max_output", &OpSchema::max_output)
    .def_property_readonly("optional_inputs", &OpSchema::optional_inputs)
    .def_property_readonly("attributes", &OpSchema::attributes)
    .def_property_readonly("input_desc", &OpSchema::input_desc)
    .def_property_readonly("output_desc", &OpSchema::output_desc)
    .def_static("is_infinite", [](int v) {
       return v == std::numeric_limits<int>::max();
     })
     .def("consumed", [](const OpSchema& schema, int i) {
       return schema.consumed(i);
     });

  py::class_<OpSchema::Attribute>(op_schema, "Attribute")
      .def_readonly("name", &OpSchema::Attribute::name)
      .def_readonly("description", &OpSchema::Attribute::description)
      .def_readonly("type", &OpSchema::Attribute::type)
      .def_readonly("required", &OpSchema::Attribute::required);

  py::enum_<OpSchema::AttrType>(op_schema, "AttrType")
      .value("FLOAT", OpSchema::AttrType::FLOAT)
      .value("INT", OpSchema::AttrType::INT)
      .value("STRING", OpSchema::AttrType::STRING)
      .value("TENSOR", OpSchema::AttrType::TENSOR)
      .value("GRAPH", OpSchema::AttrType::GRAPH)
      .value("FLOATS", OpSchema::AttrType::FLOATS)
      .value("INTS", OpSchema::AttrType::INTS)
      .value("STRINGS", OpSchema::AttrType::STRINGS)
      .value("TENSORS", OpSchema::AttrType::TENSORS)
      .value("GRAPHS", OpSchema::AttrType::GRAPHS);

  py::enum_<OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", OpSchema::SupportType::EXPERIMENTAL);

  py::enum_<OpSchema::UseType>(op_schema, "UseType")
      .value("DEFAULT", OpSchema::UseType::DEFAULT)
      .value("CONSUME_ALLOWED", OpSchema::UseType::CONSUME_ALLOWED)
      .value("CONSUME_ENFORCED", OpSchema::UseType::CONSUME_ENFORCED);

  defs.def("has_schema", [](const std::string& op_type) -> bool {
      return OpSchemaRegistry::Schema(op_type) != nullptr;
    });
  defs.def("get_schema", [](const std::string& op_type) -> OpSchema {
      const auto* schema = OpSchemaRegistry::Schema(op_type);
      if (!schema) {
        throw std::runtime_error(
          "No schema registered for '" + op_type + "'!");
      }
      return *schema;
    });

  defs.def("get_all_schemas", []() ->
        const std::unordered_map<std::string, OpSchema> {
          return OpSchemaRegistry::registered_schemas();
        });

  // Submodule `checker`
  auto checker = onnx_cpp2py_export.def_submodule("checker");
  checker.doc() = "Checker submodule";

  py::register_exception<checker::ValidationError>(checker,
                                                   "ValidationError");

  checker.def("check_value_info", [](const py::bytes& bytes,
                                     int ir_version) -> void {
          std::unique_ptr<ValueInfoProto> proto(new ValueInfoProto());
          ParseProtoFromPyBytes(proto.get(), bytes);
          checker::check_value_info(*proto, ir_version);
      });

  checker.def("check_tensor", [](const py::bytes& bytes,
                                 int ir_version) -> void {
          std::unique_ptr<TensorProto> proto(new TensorProto());
          ParseProtoFromPyBytes(proto.get(), bytes);
          checker::check_tensor(*proto, ir_version);
      });

  checker.def("check_attribute", [](const py::bytes& bytes,
                                    int ir_version) -> void {
          std::unique_ptr<AttributeProto> proto(new AttributeProto());
          ParseProtoFromPyBytes(proto.get(), bytes);
          checker::check_attribute(*proto, ir_version);
      });

  checker.def("check_node", [](const py::bytes& bytes,
                               int ir_version) -> void {
          std::unique_ptr<NodeProto> proto(new NodeProto());
          ParseProtoFromPyBytes(proto.get(), bytes);
          checker::check_node(*proto, ir_version);
      });

  checker.def("check_graph", [](const py::bytes& bytes,
                                int ir_version) -> void {
          std::unique_ptr<GraphProto> proto(new GraphProto());
          ParseProtoFromPyBytes(proto.get(), bytes);
          checker::check_graph(*proto, ir_version);
      });

  checker.def("check_model", [](const py::bytes& bytes,
                                int ir_version) -> void {
          std::unique_ptr<ModelProto> proto(new ModelProto());
          ParseProtoFromPyBytes(proto.get(), bytes);
          checker::check_model(*proto, ir_version);
      });
}

}  // namespace onnx
