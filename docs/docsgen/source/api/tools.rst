
onnx.tools
==========

.. contents::
    :local:

net_drawer
++++++++++

.. autofunction:: onnx.tools.net_drawer.GetPydotGraph

.. autofunction:: onnx.tools.net_drawer.GetOpNodeProducer

::

    from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

    pydot_graph = GetPydotGraph(
        model_onnx.graph,  # model_onnx is a ModelProto instance
        name=model_onnx.graph.name,
        rankdir="TP",
        node_producer=GetOpNodeProducer("docstring"))
    pydot_graph.write_dot("graph.dot")

update_inputs_outputs_dims
++++++++++++++++++++++++++

.. autofunction:: onnx.tools.update_model_dims.update_inputs_outputs_dims
