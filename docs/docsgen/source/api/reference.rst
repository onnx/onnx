
.. _l-reference-implementation:

onnx.reference
==============


DefaultNone
+++++++++++

.. autoclass:: onnx.reference.op_run.DefaultNone
    :members:

Inference
+++++++++

.. autoclass:: onnx.reference.ReferenceEvaluator
    :members: input_names, output_names, opsets, run

OpFunction
++++++++++

.. autoclass:: onnx.reference.op_run.OpFunction
    :members: create, eval, input, output, implicit_inputs, domain, need_context, run, make_node

OpRun
+++++

.. autoclass:: onnx.reference.op_run.OpRun
    :members: create, eval, input, output, implicit_inputs, domain, need_context, run, make_node

RuntimeTypeError
++++++++++++++++

.. autoclass:: onnx.reference.op_run.RuntimeTypeError
    :members:

SparseTensor
++++++++++++

.. autoclass:: onnx.reference.op_run.SparseTensor
    :members:
