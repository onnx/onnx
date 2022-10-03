onnx.runtime
============

.. contents::
    :local:

DefaultNone
+++++++++++

.. autoclass:: onnx.runtime.op_run.op_run.DefaultNone
    :members:

Inference
+++++++++

.. autoclass:: onnx.runtime.Inference
    :members: input_names, output_names, opsets, run

OpFunction
++++++++++

.. autoclass:: onnx.runtime.op_run.OpFunction
    :members: create, eval, input, output, local_inputs, domain, need_context, run, make_node

OpRun
+++++

.. autoclass:: onnx.runtime.op_run.OpRun
    :members: create, eval, input, output, local_inputs, domain, need_context, run, make_node

RuntimeTypeError
++++++++++++++++

.. autoclass:: onnx.runtime.op_run.RuntimeTypeError
    :members:

SparseTensor
++++++++++++

.. autoclass:: onnx.runtime.op_run.SparseTensor
    :members:
