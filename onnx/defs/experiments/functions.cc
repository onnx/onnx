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
      "A MeanVarianceNormalization Function: Perform mean variance normalization on the input tensor X");
  func.set_since_version(8);
  func.add_input("X");
  func.add_input("Pow_exponent");
  func.add_output("X_MVN");
  func.add_attribute("axes");
  NodeProto* node0 = func.add_node();
  node0->set_name("Reduced_Mean_0");
  node0->set_domain("");
  node0->set_doc_string("Caculating Reduced Mean on input tensor X");
  node0->set_op_type("ReduceMean");
  node0->add_input("X");
  node0->add_output("X_RM");
  AttributeProto* attr0 = node0->add_attribute();
  attr0->set_ref_attr_name("axes");
  attr0->set_name("axes");
  attr0->set_type(AttributeProto_AttributeType_INTS);
  NodeProto* node1 = func.add_node();
  node1->set_name("Pow_0");
  node1->set_domain("");
  node1->set_doc_string("Caculating (EX)^2");
  node1->set_op_type("Pow");
  node1->add_input("X_RM");
  node1->add_input("Pow_exponent");
  node1->add_output("EX_POW");
  NodeProto* node2 = func.add_node();
  node2->set_name("Pow_1");
  node2->set_domain("");
  node2->set_doc_string("Caculating X^2");
  node2->set_op_type("Pow");
  node2->add_input("X");
  node2->add_input("Pow_exponent");
  node2->add_output("X_POW");
  NodeProto* node3 = func.add_node();
  node3->set_name("Reduced_Mean_1");
  node3->set_domain("");
  node3->set_doc_string("Caculating E(X^2)");
  node3->set_op_type("ReduceMean");
  node3->add_input("X_POW");
  node3->add_output("E_XPOW");
  AttributeProto* attr1 = node3->add_attribute();
  attr1->set_ref_attr_name("axes");
  attr1->set_name("axes");
  attr1->set_type(AttributeProto_AttributeType_INTS);
  NodeProto* node4 = func.add_node();
  node4->set_name("SUB_0");
  node4->set_domain("");
  node4->set_doc_string("Caculating variance (E(X^2)-(EX)^2)");
  node4->set_op_type("Sub");
  node4->add_input("EX_POW");
  node4->add_input("E_XPOW");
  node4->add_output("VAR");
  NodeProto* node5 = func.add_node();
  node5->set_name("SQRT_0");
  node5->set_domain("");
  node5->set_doc_string("Caculating standard variance from variance");
  node5->set_op_type("Sqrt");
  node5->add_input("VAR");
  node5->add_output("STD_VAR");
  NodeProto* node6 = func.add_node();
  node6->set_name("SUB_1");
  node6->set_domain("");
  node6->set_doc_string("Caculating X-EX");
  node6->set_op_type("Sub");
  node6->add_input("X");
  node6->add_input("X_RM");
  node6->add_output("X_VAR");
  NodeProto* node7 = func.add_node();
  node7->set_name("DIV_0");
  node7->set_domain("");
  node7->set_doc_string("Caculating MVN-ed tensor for output");
  node7->set_op_type("Div");
  node7->add_input("X_VAR");
  node7->add_input("STD_VAR");
  node7->add_output("X_MVN");

  return Status::OK();
}

ONNX_FUNCTION(FunctionBuilder().SetDomain("").SetBuildFunction(BuildMVN));
