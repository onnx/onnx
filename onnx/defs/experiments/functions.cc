// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/common/constants.h"
#include "onnx/common/model_helpers.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;
using SupportType = ONNX_NAMESPACE::OpSchema::SupportType;

static std::vector<NodeProto> BuildMVNFunctionBody() {
	std::vector<NodeProto> function_nodes;

  NodeProto initial_node0;
  BuildNode(
      "Pow_exponent_0",
      ONNX_DOMAIN,
      "Initialize a Constant tensor to calculate squared products",
      "Constant",
      std::vector<std::string>{},
      std::vector<std::string>{"Exponent"},
      &initial_node0);
  AttributeProto* value_attr_0 = initial_node0.add_attribute();
  value_attr_0->set_name("value");
  value_attr_0->set_doc_string(
      "Exponent (default to 2.0) to element-wisely calculate the square of a tensor");
  value_attr_0->set_type(AttributeProto_AttributeType_TENSOR);
  TensorProto* tensor_proto_0 = value_attr_0->mutable_t();
  tensor_proto_0->set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto_0->add_float_data(2.0); // [2.0]
  function_nodes.emplace_back(initial_node0);

  NodeProto initial_node1;
  BuildNode(
      "Div_epsilon_0",
      ONNX_DOMAIN,
      "Initialize a Constant tensor as epsilon to avoid division by 0",
      "Constant",
      std::vector<std::string>{},
      std::vector<std::string>{"Epsilon"},
      &initial_node1);
  AttributeProto* value_attr_1 = initial_node1.add_attribute();
  value_attr_1->set_name("value");
  value_attr_1->set_doc_string(
      "Epsilon (default to 1e-9) to element-wisely add to the divisor tensor");
  value_attr_1->set_type(AttributeProto_AttributeType_TENSOR);
  TensorProto* tensor_proto_1 = value_attr_1->mutable_t();
  tensor_proto_1->set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto_1->add_float_data((float)1e-9); // [1e-9]
  function_nodes.emplace_back(initial_node1);

  NodeProto node0;
  BuildNode(
      "Reduced_Mean_0",
      ONNX_DOMAIN,
      "Calculate Reduced Mean on input tensor X",
      "ReduceMean",
      std::vector<std::string>{"X"},
      std::vector<std::string>{"X_RM"},
      &node0);
  AttributeProto* attr0 = node0.add_attribute();
  attr0->set_ref_attr_name("axes");
  attr0->set_name("axes");
  attr0->set_type(AttributeProto_AttributeType_INTS);
  function_nodes.emplace_back(node0);

  NodeProto node1;
  BuildNode(
      "Pow_0",
      ONNX_DOMAIN,
      "Calculate (EX)^2",
      "Pow",
      std::vector<std::string>{"X_RM", "Exponent"},
      std::vector<std::string>{"EX_squared"},
      &node1);
  function_nodes.emplace_back(node1);

  NodeProto node2;
  BuildNode(
      "Pow_1",
      ONNX_DOMAIN,
      "Calculate X^2",
      "Pow",
      std::vector<std::string>{"X", "Exponent"},
      std::vector<std::string>{"X_squared"},
      &node2);
  function_nodes.emplace_back(node2);

  NodeProto node3;
  BuildNode(
      "Reduced_Mean_1",
      ONNX_DOMAIN,
      "Calculate E(X^2)",
      "ReduceMean",
      std::vector<std::string>{"X_squared"},
      std::vector<std::string>{"E_Xsquared"},
      &node3);
  AttributeProto* attr1 = node3.add_attribute();
  attr1->set_ref_attr_name("axes");
  attr1->set_name("axes");
  attr1->set_type(AttributeProto_AttributeType_INTS);
  function_nodes.emplace_back(node3);

  NodeProto node4;
  BuildNode(
      "SUB_0",
      ONNX_DOMAIN,
      "Calculate variance (E(X^2)-(EX)^2)",
      "Sub",
      std::vector<std::string>{"E_Xsquared", "EX_squared"},
      std::vector<std::string>{"Variance"},
      &node4);
  function_nodes.emplace_back(node4);

  NodeProto node5;
  BuildNode(
      "SQRT_0",
      ONNX_DOMAIN,
      "Calculate standard variance from variance",
      "Sqrt",
      std::vector<std::string>{"Variance"},
      std::vector<std::string>{"STD"},
      &node5);
  function_nodes.emplace_back(node5);

  NodeProto node6;
  BuildNode(
      "SUB_1",
      ONNX_DOMAIN,
      "Calculate X-EX",
      "Sub",
      std::vector<std::string>{"X", "X_RM"},
      std::vector<std::string>{"X_variance"},
      &node6);
  function_nodes.emplace_back(node6);

  NodeProto node7;
  BuildNode(
      "ADD_0",
      ONNX_DOMAIN,
      "Add epsilon value to STD to avoid division by 0",
      "Add",
      std::vector<std::string>{"STD", "Epsilon"},
      std::vector<std::string>{"Processed_STD"},
      &node7);
  function_nodes.emplace_back(node7);

  NodeProto node8;
  BuildNode(
      "DIV_0",
      ONNX_DOMAIN,
      "Calculate MVN-ed tensor for output",
      "Div",
      std::vector<std::string>{"X_variance", "Processed_STD"},
      std::vector<std::string>{"Y"},
      &node8);
  function_nodes.emplace_back(node8);

  return function_nodes;
}

static const char* mvn_ver9_doc = R"DOC(
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    9,
    OpSchema()
        .SetSupportLevel(SupportType::COMMON)
        .SetDoc(mvn_ver9_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Attr(
            "axes",
			"A list of integers, along which to reduce. The default is to reduce over "
			"all the dimensions of the input tensor. Use [0,2,3] (without C axis for "
            "N-D cases) for calculating means and variances along channels. Two "
            "variables with the same C-coordinate are associated "
            "with the same mean and variance.",
			AttributeProto::INTS,
			OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to all numeric tensors.")
       .FunctionBody(BuildMVNFunctionBody()));

} // namespace ONNX_NAMESPACE
