.. _l-npx:

Numpy API for ONNX
==================

See `Python array API standard <https://data-apis.org/array-api/latest/index.html>`_.

.. contents::
    :local:

Functions
+++++++++

.. autofunction:: onnx.npx.npx_functions.abs

.. autofunction:: onnx.npx.npx_functions.absolute

.. autofunction:: onnx.npx.npx_functions.arccos

.. autofunction:: onnx.npx.npx_functions.arccosh

.. autofunction:: onnx.npx.npx_functions.amax

.. autofunction:: onnx.npx.npx_functions.amin

.. autofunction:: onnx.npx.npx_functions.arange

.. autofunction:: onnx.npx.npx_functions.argmax

.. autofunction:: onnx.npx.npx_functions.argmin

.. autofunction:: onnx.npx.npx_functions.arcsin

.. autofunction:: onnx.npx.npx_functions.arcsinh

.. autofunction:: onnx.npx.npx_functions.arctan

.. autofunction:: onnx.npx.npx_functions.arctanh

.. autofunction:: onnx.npx.npx_functions.cdist

.. autofunction:: onnx.npx.npx_functions.ceil

.. autofunction:: onnx.npx.npx_functions.clip

.. autofunction:: onnx.npx.npx_functions.compress

.. autofunction:: onnx.npx.npx_functions.compute

.. autofunction:: onnx.npx.npx_functions.concat

.. autofunction:: onnx.npx.npx_functions.cos

.. autofunction:: onnx.npx.npx_functions.cosh

.. autofunction:: onnx.npx.npx_functions.cumsum

.. autofunction:: onnx.npx.npx_functions.det

.. autofunction:: onnx.npx.npx_functions.dot

.. autofunction:: onnx.npx.npx_functions.einsum

.. autofunction:: onnx.npx.npx_functions.erf

.. autofunction:: onnx.npx.npx_functions.exp

.. autofunction:: onnx.npx.npx_functions.expand_dims

.. autofunction:: onnx.npx.npx_functions.expit

.. autofunction:: onnx.npx.npx_functions.floor

.. autofunction:: onnx.npx.npx_functions.hstack

.. autofunction:: onnx.npx.npx_functions.copy

.. autofunction:: onnx.npx.npx_functions.identity

.. autofunction:: onnx.npx.npx_functions.isnan

.. autofunction:: onnx.npx.npx_functions.log

.. autofunction:: onnx.npx.npx_functions.log1p

.. autofunction:: onnx.npx.npx_functions.matmul

.. autofunction:: onnx.npx.npx_functions.pad

.. autofunction:: onnx.npx.npx_functions.reciprocal

.. autofunction:: onnx.npx.npx_functions.relu

.. autofunction:: onnx.npx.npx_functions.round

.. autofunction:: onnx.npx.npx_functions.sigmoid

.. autofunction:: onnx.npx.npx_functions.sign

.. autofunction:: onnx.npx.npx_functions.sin

.. autofunction:: onnx.npx.npx_functions.sinh

.. autofunction:: onnx.npx.npx_functions.squeeze

.. autofunction:: onnx.npx.npx_functions.tan

.. autofunction:: onnx.npx.npx_functions.tanh

.. autofunction:: onnx.npx.npx_functions.topk

.. autofunction:: onnx.npx.npx_functions.transpose

.. autofunction:: onnx.npx.npx_functions.unsqueeze

.. autofunction:: onnx.npx.npx_functions.vstack

.. autofunction:: onnx.npx.npx_functions.where

Var
+++

.. autofunction:: onnx.npx.npx_var.Var

Cst, Input
++++++++++

.. autofunction:: onnx.npx.npx_var.Cst

.. autofunction:: onnx.npx.npx_var.Input

API
+++

.. autofunction:: onnx.npx.npx_core_api.var

.. autofunction:: onnx.npx.npx_core_api.cst

.. autofunction:: onnx.npx.npx_jit_eager.jit_eager

.. autofunction:: onnx.npx.npx_jit_eager.jit_onnx

.. autofunction:: onnx.npx.npx_core_api.make_tuple

.. autofunction:: onnx.npx.npx_core_api.tuple_var

.. autofunction:: onnx.npx.npx_core_api.npxapi_inline

.. autofunction:: onnx.npx.npx_core_api.npxapi_function

JIT, Eager
++++++++++

.. autofunction:: onnx.npx.npx_jit_eager.JitEager

.. autofunction:: onnx.npx.npx_jit_eager.JitOnnx

Tensors
+++++++

.. autofunction:: onnx.npx.npx_tensors.NumpyTensor

Annotations
+++++++++++

.. autofunction:: onnx.npx.npx_types.ElemType

.. autofunction:: onnx.npx.npx_types.ParType

.. autofunction:: onnx.npx.npx_types.OptParType

.. autofunction:: onnx.npx.npx_types.TensorType

.. autofunction:: onnx.npx.npx_types.SequenceType

.. autofunction:: onnx.npx.npx_types.TupleType

.. autofunction:: onnx.npx.npx_types.Bool

.. autofunction:: onnx.npx.npx_types.BFloat16

.. autofunction:: onnx.npx.npx_types.Float16

.. autofunction:: onnx.npx.npx_types.Float32

.. autofunction:: onnx.npx.npx_types.Float64

.. autofunction:: onnx.npx.npx_types.Int8

.. autofunction:: onnx.npx.npx_types.Int16

.. autofunction:: onnx.npx.npx_types.Int32

.. autofunction:: onnx.npx.npx_types.Int64

.. autofunction:: onnx.npx.npx_types.UInt8

.. autofunction:: onnx.npx.npx_types.UInt16

.. autofunction:: onnx.npx.npx_types.UInt32

.. autofunction:: onnx.npx.npx_types.UInt64
