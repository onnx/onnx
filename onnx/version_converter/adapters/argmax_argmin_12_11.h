// Adapter for ArgMax/ArgMin in default domain from version 12 to 11

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class ArgMaxArgMin_12_11 final : public Adapter {
  public:
    explicit ArgMaxArgMin_12_11(const std::string& op_name)
      : Adapter(op_name, OpSetID(12), OpSetID(11)) {}

    void adapt_argmax_argmin_12_11(std::shared_ptr<Graph> graph, Node* node) const {
    	Symbol select_last_index = Symbol("select_last_index");
	    if (node->hasAttribute(select_last_index)) {
	      ONNX_ASSERTM(node->i(select_last_index) == 0,
	          "select_last_index != 0 is not supported in version 11");
	      node->removeAttribute(select_last_index);
	    }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
	    adapt_argmax_argmin_12_11(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
