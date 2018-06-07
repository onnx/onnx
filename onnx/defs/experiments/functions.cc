// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
using namespace ONNX_NAMESPACE;

static Common::Status BuildFc(std::unique_ptr<FunctionProto>* func_proto) {
  if (nullptr == func_proto) {
    return Status(
      Common::CHECKER,
      Common::INVALID_ARGUMENT,
      "func_proto should not be nullptr.");
  }

  func_proto->reset(new FunctionProto);
  auto& func = **func_proto;
  func.set_name("FC");
  func.set_doc_string("this is a full connection function.");
  func.set_since_version(8);
  func.add_input("w");
  func.add_input("x");
  func.add_input("b");
  func.add_output("y");
  NodeProto* node0 = func.add_node();
  node0->set_name("node0");
  node0->set_domain("");
  node0->set_doc_string("This is a matmul testing node ");
  node0->set_op_type("MatMul");
  node0->add_input("w");
  node0->add_input("x");
  node0->add_output("y_1");
  NodeProto* node1 = func.add_node();
  node1->set_name("node1");
  node1->set_domain("");
  node1->set_doc_string("This is a add testing node ");
  node1->set_op_type("Add");
  node1->add_input("y_1");
  node1->add_input("b");
  node1->add_output("y");
  
  //set function inputs.
  //set function outputs.
  //set function attributes.
  //set function description.
  //set function body (nodes).

  return Status::OK();
}

ONNX_FUNCTION(FunctionBuilder().SetDomain("").SetBuildFunction(BuildFc));
