onnx.funconnx
=============

.. contents::
    :local:

DefaultNone
+++++++++++

.. autoclass:: onnx.funconnx.op_run.DefaultNone
    :members:

Inference
+++++++++

.. autoclass:: onnx.funconnx.ProtoRun
    :members: input_names, output_names, opsets, run

OpFunction
++++++++++

.. autoclass:: onnx.funconnx.op_run.OpFunction
    :members: create, eval, input, output, implicit_inputs, domain, need_context, run, make_node

OpRun
+++++

.. autoclass:: onnx.funconnx.op_run.OpRun
    :members: create, eval, input, output, implicit_inputs, domain, need_context, run, make_node

RuntimeTypeError
++++++++++++++++

.. autoclass:: onnx.funconnx.op_run.RuntimeTypeError
    :members:

SparseTensor
++++++++++++

.. autoclass:: onnx.funconnx.op_run.SparseTensor
    :members:
