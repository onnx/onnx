#include <pybind11/pybind11.h>
#include <climits>
#include <limits>
#include <pybind11/stl.h>
#include <unordered_map>

#include "onnx/defs/schema.h"

namespace onnx {

namespace py = pybind11;


PYBIND11_PLUGIN(onnx_cpp2py_export) {
  py::module m(
      "onnx_cpp2py_export",
      "Python interface to onnx schemas");

  // OpSchema
  py::class_<OpSchema> op_schema(m, "OpSchema");
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
    .def_property_readonly("attributes", &OpSchema::attributes)
    .def_property_readonly("inputs", &OpSchema::inputs)
    .def_property_readonly("outputs", &OpSchema::outputs)
    .def_property_readonly("type_constraints", &OpSchema::typeConstraintParams)
    .def("verify", [](const OpSchema& schema,
                      const py::bytes& serialized_node_proto) -> bool {
       std::unique_ptr<NodeProto> node_proto(new NodeProto());
       node_proto->ParseFromString(serialized_node_proto);
       return schema.Verify(*node_proto);
     })
     .def_static("is_infinite", [](int v) {
       return v == std::numeric_limits<int>::max();
     })
     .def("consumed", [](const OpSchema& schema, int i) {
       return schema.consumed(i);
     });

  //
  py::class_<OpSchema::Attribute>(op_schema, "Attribute")
      .def_readonly("name", &OpSchema::Attribute::name)
      .def_readonly("description", &OpSchema::Attribute::description)
      .def_readonly("type", &OpSchema::Attribute::type)
      .def_readonly("required", &OpSchema::Attribute::required);
      
  py::class_<OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def_property_readonly("name", &OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("typeStr", &OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly("description", &OpSchema::FormalParameter::GetDescription);

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

  m.def("get_schema", [](const std::string& op_type) -> OpSchema {
      const auto* schema = OpSchemaRegistry::Schema(op_type);
      if (!schema) {
        throw std::runtime_error(
          "No schema registered for '" + op_type + "'!");
      }
      return *schema;
    });

  m.def("has_schema", [](const std::string& op_type) -> bool {
      return OpSchemaRegistry::Schema(op_type) != nullptr;
    });

  m.def("get_all_schemas", []() ->
        const std::unordered_map<std::string, OpSchema> {
          return OpSchemaRegistry::registered_schemas();
        });

  m.def("is_attribute_legal", [](const py::bytes& serialized_attr_proto) -> bool {
       std::unique_ptr<AttributeProto> attr_proto(new AttributeProto());
       attr_proto->ParseFromString(serialized_attr_proto);
       return OpSchema::IsAttributeLegal(*attr_proto);
     });

  return m.ptr();
}

}  // namespace onnx
