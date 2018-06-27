// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
using namespace ONNX_NAMESPACE;

static Common::Status BuildMVN(std::unique_ptr<FunctionProto>* func_proto) {
  if (nullptr == func_proto) {
    return Status(
        Common::CHECKER,
        Common::INVALID_ARGUMENT,
        "func_proto should not be nullptr.");
  }

  func_proto->reset(new FunctionProto);
  auto& func = **func_proto;
  func.set_name("MeanVarianceNormalization");
  func.set_doc_string(
      "A MeanVarianceNormalization Function: Perform mean variance normalization "
      "on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ``` <br/>"
      "INPUT: X(float/float16/double) with Shape [N,C,W,H] <br/>"
      "ATTRIBUTE: <br/>&nbsp;&nbsp;&nbsp;&nbsp;axes: will be passed to ReducedMean Ops. "
      "Use [0,2,3] for no across channel, [0,1,2,3] for across channel caculation.<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;(The KeepDims attribute in ReducedMean ops is set to true for caculation)<br/>"
      "OUTPUT: X(float/float16/double) with Shape [N,C,W,H] <br/>");
  func.set_since_version(8);
  func.add_input("X");
  func.add_output("X_MVN");
  func.add_attribute("axes");
  NodeProto* initial_node0 = func.add_node();
  FunctionNewNodeHelper(
      initial_node0,
      "Pow_exponent_0",
      "",
      "Initialize a Constant tensor to caculate squared products",
      "Constant",
      std::vector<std::string>{},
      std::vector<std::string>{});
  AttributeProto* value_attr = initial_node0->add_attribute();
  value_attr->set_name("value");
  value_attr->set_doc_string(
      "Value for the constant tensor as power exponent to caculate MVN");
  value_attr->set_type(AttributeProto_AttributeType_TENSOR);
  TensorProto* tensor_proto = value_attr->mutable_t();
  tensor_proto->set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto->add_float_data(2.0); // [2.0]
  initial_node0->add_output("Exponent");
  NodeProto* node0 = func.add_node();
  FunctionNewNodeHelper(
      node0,
      "Reduced_Mean_0",
      "",
      "Caculate Reduced Mean on input tensor X",
      "ReduceMean",
      std::vector<std::string>{"X"},
      std::vector<std::string>{"X_RM"});
  AttributeProto* attr0 = node0->add_attribute();
  attr0->set_ref_attr_name("axes");
  attr0->set_name("axes");
  attr0->set_type(AttributeProto_AttributeType_INTS);
  NodeProto* node1 = func.add_node();
  FunctionNewNodeHelper(
      node1,
      "Pow_0",
      "",
      "Caculate (EX)^2",
      "Pow",
      std::vector<std::string>{"X_RM", "Exponent"},
      std::vector<std::string>{"EX_squared"});
  NodeProto* node2 = func.add_node();
  FunctionNewNodeHelper(
      node2,
      "Pow_1",
      "",
      "Caculate X^2",
      "Pow",
      std::vector<std::string>{"X", "Exponent"},
      std::vector<std::string>{"X_squared"});
  NodeProto* node3 = func.add_node();
  FunctionNewNodeHelper(
      node3,
      "Reduced_Mean_1",
      "",
      "Caculate E(X^2)",
      "ReduceMean",
      std::vector<std::string>{"X_squared"},
      std::vector<std::string>{"E_Xsquared"});
  AttributeProto* attr1 = node3->add_attribute();
  attr1->set_ref_attr_name("axes");
  attr1->set_name("axes");
  attr1->set_type(AttributeProto_AttributeType_INTS);
  NodeProto* node4 = func.add_node();
  FunctionNewNodeHelper(
      node4,
      "SUB_0",
      "",
      "Caculate variance (E(X^2)-(EX)^2)",
      "Sub",
      std::vector<std::string>{"EX_squared", "E_Xsquared"},
      std::vector<std::string>{"Variance"});
  NodeProto* node5 = func.add_node();
  FunctionNewNodeHelper(
      node5,
      "SQRT_0",
      "",
      "Caculate standard variance from variance",
      "Sqrt",
      std::vector<std::string>{"Variance"},
      std::vector<std::string>{"STD"});
  NodeProto* node6 = func.add_node();
  FunctionNewNodeHelper(
      node6,
      "SUB_1",
      "",
      "Caculate X-EX",
      "Sub",
      std::vector<std::string>{"X", "X_RM"},
      std::vector<std::string>{"X_variance"});
  NodeProto* node7 = func.add_node();
  FunctionNewNodeHelper(
      node7,
      "DIV_0",
      "",
      "Caculate MVN-ed tensor for output",
      "Div",
      std::vector<std::string>{"X_variance", "STD"},
      std::vector<std::string>{"X_MVN"});

  return Status::OK();
}

ONNX_FUNCTION(FunctionBuilder().SetDomain("").SetBuildFunction(BuildMVN));
