# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object,unnecessary-lambda,raise-missing-from,unidiomatic-typecheck,import-outside-toplevel,ungrouped-imports,reimported

import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from numpy.testing import assert_allclose

from onnx import FunctionProto, ModelProto, TensorProto
from onnx.backend.test.case.node.pad import pad_impl
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.npx import ElemType, eager_onnx, jit_onnx
from onnx.npx.npx_core_api import cst, make_tuple, npxapi_function, npxapi_inline
from onnx.npx.npx_functions import absolute as absolute_inline
from onnx.npx.npx_functions import arange as arange_inline
from onnx.npx.npx_functions import arccos as arccos_inline
from onnx.npx.npx_functions import arccosh as arccosh_inline
from onnx.npx.npx_functions import arcsin as arcsin_inline
from onnx.npx.npx_functions import arcsinh as arcsinh_inline
from onnx.npx.npx_functions import arctan as arctan_inline
from onnx.npx.npx_functions import arctanh as arctanh_inline
from onnx.npx.npx_functions import argmin as argmin_inline
from onnx.npx.npx_functions import cdist as cdist_inline
from onnx.npx.npx_functions import ceil as ceil_inline
from onnx.npx.npx_functions import clip as clip_inline
from onnx.npx.npx_functions import compress as compress_inline
from onnx.npx.npx_functions import compute as compute_inline
from onnx.npx.npx_functions import concat as concat_inline
from onnx.npx.npx_functions import copy as copy_inline
from onnx.npx.npx_functions import cos as cos_inline
from onnx.npx.npx_functions import cosh as cosh_inline
from onnx.npx.npx_functions import cumsum as cumsum_inline
from onnx.npx.npx_functions import det as det_inline
from onnx.npx.npx_functions import dot as dot_inline
from onnx.npx.npx_functions import einsum as einsum_inline
from onnx.npx.npx_functions import erf as erf_inline
from onnx.npx.npx_functions import exp as exp_inline
from onnx.npx.npx_functions import expand_dims as expand_dims_inline
from onnx.npx.npx_functions import expit as expit_inline
from onnx.npx.npx_functions import floor as floor_inline
from onnx.npx.npx_functions import hstack as hstack_inline
from onnx.npx.npx_functions import identity as identity_inline
from onnx.npx.npx_functions import isnan as isnan_inline
from onnx.npx.npx_functions import log as log_inline
from onnx.npx.npx_functions import log1p as log1p_inline
from onnx.npx.npx_functions import matmul as matmul_inline
from onnx.npx.npx_functions import pad as pad_inline
from onnx.npx.npx_functions import reciprocal as reciprocal_inline
from onnx.npx.npx_functions import relu as relu_inline
from onnx.npx.npx_functions import round as round_inline
from onnx.npx.npx_functions import sigmoid as sigmoid_inline
from onnx.npx.npx_functions import sign as sign_inline
from onnx.npx.npx_functions import sin as sin_inline
from onnx.npx.npx_functions import sinh as sinh_inline
from onnx.npx.npx_functions import sqrt as sqrt_inline
from onnx.npx.npx_functions import squeeze as squeeze_inline
from onnx.npx.npx_functions import tan as tan_inline
from onnx.npx.npx_functions import tanh as tanh_inline
from onnx.npx.npx_functions import topk as topk_inline
from onnx.npx.npx_functions import transpose as transpose_inline
from onnx.npx.npx_functions import unsqueeze as unsqueeze_inline
from onnx.npx.npx_functions import vstack as vstack_inline
from onnx.npx.npx_functions import where as where_inline
from onnx.npx.npx_functions_test import (
    _min_max,
    _min_max_inline,
    absolute,
    addition,
    argmin,
    concat,
    copy,
    log1p,
    negative,
    relu,
    topk,
)
from onnx.npx.npx_types import Bool, Float32, Float64, Int64, OptParType, TensorType
from onnx.npx.npx_var import Input, Var
from onnx.reference import ReferenceEvaluator
from onnx.shape_inference import infer_shapes

try:
    from onnxruntime import InferenceSession
except ImportError:
    InferenceSession = None

try:
    import scipy
except ImportError:
    scipy = None

if InferenceSession is not None:
    # This is an example of a backend for classes JitOnnx and JitEager
    # using onnxruntime as a runtime. It is provided as an example.
    from typing import Any, Callable, List, Optional, Tuple, Union

    import numpy as np
    from onnxruntime import InferenceSession, RunOptions
    from onnxruntime.capi._pybind_state import OrtDevice as C_OrtDevice
    from onnxruntime.capi._pybind_state import OrtMemType
    from onnxruntime.capi._pybind_state import (
        OrtValue as C_OrtValue,  # pylint: disable=E0611
    )
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    from onnx import ModelProto, TensorProto
    from onnx.defs import onnx_opset_version
    from onnx.npx.npx_tensors import BackendTensor, EagerTensor
    from onnx.npx.npx_types import TensorType

    class OrtTensor:
        """
        Default backend based on
        :class:`onnxruntime.InferenceSession`.
        Data is not copied.

        :param input_names: input names
        :param onx: onnx model
        """

        CPU = C_OrtDevice(C_OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        CUDA0 = C_OrtDevice(C_OrtDevice.cuda(), OrtMemType.DEFAULT, 0)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def from_array(
            value: np.ndarray, device: Optional[C_OrtDevice] = None
        ) -> "OrtTensor":
            """
            Creates an instance of :class:`OrtTensor` from a numpy array.
            Relies on `ortvalue_from_numpy`.
            A copy of the data in the Numpy object is held by the
            :class:`C_OrtValue` only if the device is **not cpu**.
            Any expression such as `from_array(x.copy())`, or
            `from_array(x.astype(np.float32))`, ... creates an intermediate
            variable scheduled to be deleted by the garbage collector
            as soon as the function returns. In that case, the buffer
            holding the values is deleted and the instance `OrtTenor`
            is no longer equal to the original value:
            `assert_allclose(value, tensor.numpy())` is false.
            `value` must remain alive as long as the `OrtTensor` is.

            :param value: value
            :param device: CPU, GPU, value such as `OrtTensor.CPU`,
                `OrtTensor.CUDA0`
            :return: instance of :class:`OrtTensor`
            """
            if device is None:
                device = OrtTensor.CPU
            return OrtTensor(C_OrtValue.ortvalue_from_numpy(value, device))

        def numpy(self) -> np.ndarray:
            """
            Converts the :class:`OrtValue` into numpy array.
            """
            return self._tensor.numpy()  # type: ignore[no-any-return]

        class Evaluator:
            """
            Wraps class :class:`onnxruntime.InferenceSession`
            to have a signature closer to python function.
            """

            def __init__(
                self, tensor_class: type, input_names: List[str], onx: ModelProto
            ):
                try:
                    self.ref = InferenceSession(
                        onx.SerializeToString(),  # type: ignore[attr-defined]
                        providers=tensor_class.providers,  # type: ignore[attr-defined]
                    )
                except InvalidArgument as e:
                    if (
                        len(onx.graph.output) == 1
                        and onx.graph.output[0].type.tensor_type.elem_type
                        == TensorProto.UNDEFINED
                    ):
                        # ShapeInference cannot use python function for unknown node type.
                        # Let's give the only output the same type as the first input.
                        onx.graph.output[
                            0
                        ].type.tensor_type.elem_type = onx.graph.input[
                            0
                        ].type.tensor_type.elem_type
                        self.ref = InferenceSession(
                            onx.SerializeToString(),  # type: ignore[attr-defined]
                            providers=tensor_class.providers,  # type: ignore[attr-defined]
                        )
                    else:
                        if len(onx.graph.node) <= 3:
                            raise RuntimeError(
                                f"Unable to create an InferenceSession with model {onx}."
                            ) from e
                        raise e
                self.input_names = input_names
                self.tensor_class = tensor_class
                self.output_names = [output.name for output in self.ref._outputs_meta]
                self.run_options = RunOptions()

            def run(self, *inputs: List["OrtTensor"]) -> List["OrtTensor"]:
                """
                Executes the function.

                :param inputs: function inputs
                :return: outputs
                """
                if len(inputs) != len(self.input_names):
                    raise ValueError(
                        f"Expected {len(self.input_names)} inputs but got "
                        f"len(inputs)={len(inputs)}."
                    )
                feeds = {}
                for name, inp in zip(self.input_names, inputs):
                    feeds[name] = inp.value  # type: ignore[attr-defined]
                res = self.ref._sess.run_with_ort_values(  # pylint: disable=protected-access
                    feeds, self.output_names, self.run_options
                )
                return list(map(OrtTensor, res))

        def __init__(self, tensor: Union[C_OrtValue, "OrtTensor"]):
            if isinstance(tensor, C_OrtValue):
                self._tensor = tensor
            elif isinstance(tensor, OrtTensor):
                self._tensor = tensor._tensor
            else:
                raise ValueError(f"An OrtValue is expected not {type(tensor)}.")

        @property
        def shape(self) -> Tuple[int, ...]:
            "Returns the shape of the tensor."
            return self._tensor.shape()  # type: ignore[no-any-return]

        @property
        def dtype(self) -> Any:
            "Returns the element type of this tensor."
            return self._tensor.element_type()

        @property
        def key(self) -> Any:
            "Unique key for a tensor of the same type."
            return (self.dtype, len(self.shape))

        @property
        def value(self) -> C_OrtValue:
            "Returns the value of this tensor as a numpy array."
            return self._tensor

        @property
        def tensor_type(self) -> TensorType:
            "Returns the tensor type of this tensor."
            return TensorType[self.dtype]  # type: ignore[misc,no-any-return]

        @property
        def dims(self):
            """
            Returns the dimensions of the tensor.
            First dimension is the batch dimension if the tensor
            has more than one dimension.
            """
            if len(self.shape) == 0:
                return (0,)
            if len(self.shape) == 1:
                return tuple(self.shape)
            return (None, *tuple(self.shape[1:]))

        @property
        def tensor_type_dims(self) -> TensorType:
            """
            Returns the tensor type of this tensor.
            This property is used to define a key used to cache a jitted function.
            Same keys keys means same ONNX graph.
            Different keys usually means same ONNX graph but different
            input shapes.
            """
            return TensorType[self.dtype, self.dims]  # type: ignore[misc,no-any-return]

        @classmethod
        def create_function(
            cls: Any, input_names: List[str], onx: ModelProto
        ) -> Callable:
            """
            Creates a python function calling the onnx backend
            used by this class.

            :param onx: onnx model
            :return: python function
            """
            return cls.Evaluator(cls, input_names, onx)  # type: ignore[return-value]

    class BackendOrtTensor(OrtTensor, BackendTensor):
        """
        Defines a value for a specific backend.
        """

        @classmethod
        def get_opsets(cls, opsets):
            if opsets is None:
                return {"": onnx_opset_version(), "com.microsoft": 1}
            if "com.microsoft" in opsets:
                return opsets
            opsets = opsets.copy()
            opsets.update({"com.microsoft": 1})
            return opsets

        @classmethod
        def get_ir_version(cls, ir_version):
            return ir_version

    class EagerOrtTensor(OrtTensor, EagerTensor):
        """
        Defines a value for a specific backend.
        """

        @classmethod
        def get_opsets(cls, opsets):
            if opsets is None:
                return {"": onnx_opset_version(), "com.microsoft": 1}
            if "com.microsoft" in opsets:
                return opsets
            opsets = opsets.copy()
            opsets.update({"com.microsoft": 1})
            return opsets

        @classmethod
        def get_ir_version(cls, ir_version):
            return ir_version


DEFAULT_OPSET = onnx_opset_version()


class TestNpx(unittest.TestCase):
    _warns = []  # type: ignore[var-annotated]

    def assertEqualArray(self, expected, value, atol=0, rtol=0):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertRaise(self, fct, exc_type):
        try:
            fct()
            e = None
        except exc_type as e:
            if type(e) != exc_type:  # pylint: disable=unidiomatic-typecheck
                raise AssertionError(f"Unexpected exception {type(e)!r}.")
            return
        if e is None:
            raise AssertionError("No exception was raised.")
        raise AssertionError(f"Unexpected exception {type(e)!r}.")

    def assertEmpty(self, value):
        if value is None:
            return
        if len(value) == 0:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if len(value) == 0:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix, full):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for w in TestNpx._warns:
            warnings.warn(w)

    def test_shape_inference(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.UNDEFINED, [None, None])
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])
        graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        shapes = infer_shapes(onnx_model)
        output = shapes.graph.output[0]
        self.assertEqual(output.type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_tensor(self):
        dt = TensorType["float32"]  # type: ignore[misc,name-defined,type-arg]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEmpty(dt.shape)
        self.assertEqual(dt.type_name(), "TensorType['float32']")
        dt = TensorType["float32"]  # type: ignore[misc,name-defined]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "TensorType['float32']")
        dt = TensorType[np.float32]  # type: ignore[misc,name-defined]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "TensorType['float32']")
        self.assertEmpty(dt.shape)

        self.assertRaise(lambda: TensorType[None], TypeError)  # type: ignore[misc]
        self.assertRaise(lambda: TensorType[np.str_], TypeError)  # type: ignore[misc]
        self.assertRaise(lambda: TensorType[{np.float32, np.str_}], TypeError)  # type: ignore[misc]

    def test_superset(self):
        t1 = TensorType[ElemType.numerics]  # type: ignore[type-arg,valid-type]
        t2 = TensorType[ElemType.float64]  # type: ignore[type-arg,valid-type]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[None]  # type: ignore[misc]
        t2 = Float32[None]  # type: ignore[misc]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[5]  # type: ignore[misc]
        t2 = Float32[5]  # type: ignore[misc]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[None]  # type: ignore[misc]
        t2 = Float32[5]  # type: ignore[misc]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32["N"]  # type: ignore[misc]
        t2 = Float32[5]  # type: ignore[misc]
        self.assertTrue(t1.issuperset(t2))
        t1 = TensorType[ElemType.int64]  # type: ignore[misc]
        t2 = Int64[1]  # type: ignore[misc]
        self.assertTrue(t1.issuperset(t2))

    def test_sig(self):
        def local1(
            x: TensorType[ElemType.floats],  # type: ignore[type-arg,valid-type]
        ) -> TensorType[ElemType.floats]:  # type: ignore[name-defined,type-arg,valid-type]
            return x

        def local2(
            x: TensorType[ElemType.floats, "T"]  # type: ignore[type-arg,valid-type,name-defined]
        ) -> TensorType[ElemType.floats, "T"]:  # type: ignore[name-defined,type-arg,valid-type]
            return x

        def local3(x: Float32["N", 1]) -> Float32["N", 1]:  # type: ignore[name-defined]
            return x

        def local4(x: Float64["N", 1]) -> Int64["N", 1]:  # type: ignore[name-defined]
            return x

        self.assertNotEmpty(local1)
        self.assertNotEmpty(local2)
        self.assertNotEmpty(local3)
        self.assertNotEmpty(local4)

    def test_numpy_abs(self):
        f = absolute(Input())
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", absolute.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", absolute.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", absolute.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_neg(self):
        f = absolute(negative(Input()))
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(-x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_log1p(self):
        f = log1p(Input())
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([5, 6], dtype=np.float64)
        y = np.log1p(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_neg_constraint_input(self):
        f = absolute(negative(Input()))
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        self.assertRaise(lambda: f.to_onnx(), RuntimeError)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(-x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_two_inputs(self):
        f = absolute(addition(Input(), Input()))
        self.assertIsInstance(f, Var)
        self.assertIn("Signature", addition.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", addition.__doc__)
        self.assertIn("y: TensorType[numerics, 'T']", addition.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", addition.__doc__)
        self.assertRaise(lambda: f.to_onnx(), RuntimeError)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([2.5], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x, "I__1": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_parameter_argmin(self):
        f = argmin(Input())
        self.assertIsInstance(f, Var)
        self.assertIn("Signature", argmin.__doc__)
        self.assertIn("x: TensorType[numerics, 'T'],", argmin.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", argmin.__doc__)
        self.assertIn("axis: OptParType[int],", argmin.__doc__)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        if DEFAULT_OPSET > 18:
            z = np.argmin(x, axis=0)
            self.assertEqualArray(z, got[0])
        else:
            # bug in onnx==1.13
            self._warns.append(
                "ReferenceEvaluator:test_numpy_parameter_argmin: "
                "axis not taken into account"
            )
            self.assertIn(0, got[0].ravel().tolist())

    def test_numpy_relu(self):
        f = relu(Input())
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        z = np.where(x >= 0, x, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2(self):
        f = concat(Input(), Input())
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        z = np.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"I__0": x1, "I__1": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2_inline(self):
        f = concat_inline(Input("A"), Input("B"))
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        z = np.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_concat1_2(self):
        f = concat(Input(), concat(Input(), Input()))
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        z = np.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onx)
        feeds = {"I__2": x1, "I__0": x2, "I__1": x3}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_concat1_2_names(self):
        f = concat(Input("A"), concat(Input("B"), Input("C")))
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        z = np.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2, "C": x3}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2_2(self):
        f = concat(
            concat(Input("A"), Input("B")), concat(Input("C"), Input("D"), Input("E"))
        )
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        x4 = np.array([[10, 20]], dtype=np.float64)
        x5 = np.array([[100, 200]], dtype=np.float64)
        z = np.vstack([x1, x2, x3, x4, x5])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2, "C": x3, "D": x4, "E": x5}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_numpy_abs_a0(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_a0_true(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={(0, True): Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_aN(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None], "r__0": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_inline(self):
        f = absolute_inline(Input())
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", absolute.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", absolute.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", absolute.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        self.assertNotIn("functions {", str(onx))
        x = np.array([-5, 6], dtype=np.float64)
        y = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"I__0": x})
        self.assertEqualArray(y, got[0])

    def test_numpy_addition_op(self):
        f = absolute(addition(copy(Input("A")), Input("B")))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"T": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_inline(self):
        f = absolute_inline(copy_inline(Input("A")) + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator(self):
        f = absolute(copy(Input("A")) + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_input_inline(self):
        f = absolute_inline(Input("A") + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_input(self):
        f = absolute(Input("A") + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    def test_backend_0(self):
        def impl(A, B):
            return absolute_inline(copy_inline(A) + B)

        f = impl(Input("A"), Input("B"))

        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x, y)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64), y.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_backend_1(self):
        def impl(A, B):
            return absolute(copy(A) + B)

        f = impl(Input("A"), Input("B"))

        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x = np.array([-5, 6], dtype=np.float64)
        y = np.array([15, -16], dtype=np.float64)
        z = np.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x, y)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64), y.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_backend_parameters(self):
        def impl(A, axis=1):
            return argmin_inline(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_backend_parameters_xapi(self):
        @npxapi_inline
        def impl(A, axis=1):
            return argmin_inline(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_backend_parameters_no_inline(self):
        def impl(A, axis=1):
            return argmin(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x}
        got = ref.run(None, feeds)
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_backend_parameters_no_inline_xapi(self):
        @npxapi_function
        def impl(
            A: TensorType[ElemType.numerics, "T"], axis: OptParType[int] = 1  # type: ignore[assignment,type-arg,valid-type,name-defined]
        ) -> TensorType[ElemType.numerics, "T"]:  # type: ignore[no-any-return,type-arg,valid-type,name-defined]
            return argmin(A, axis=axis)  # type: ignore[no-any-return]

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Int64[None]})
        x = np.array([[-5, 6], [5, -6]], dtype=np.float64)
        z0 = np.argmin(x, axis=0)
        z1 = np.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x}
        got = ref.run(None, feeds)
        self.assertEqualArray(z1, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertIsInstance(f.versions, dict)
        self.assertEqual(len(f.versions), 1)
        res = f(x, axis=0)
        self.assertEqual(len(f.versions), 2)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, np.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqual(len(f.versions), 3)
        self.assertEqualArray(z1.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        res = f(x.astype(np.int64), axis=0)
        self.assertEqual(len(f.versions), 4)
        self.assertEqualArray(z0.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

        # versions
        self.assertIsInstance(f.onxs, dict)
        self.assertEqual(len(f.onxs), 4)
        keys = list(sorted(f.onxs))
        self.assertIsInstance(f.onxs[keys[0]], ModelProto)
        k = keys[-1]
        self.assertEqual(len(k), 3)
        self.assertEqual(k[1:], ("axis", 0))

    def test_numpy_topk(self):
        f = topk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", topk.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", topk.__doc__)
        self.assertIn("k: TensorType['int64', (1,), 'I']", topk.__doc__)
        self.assertIn(
            ") -> TupleType[TensorType[numerics, 'T'], TensorType['int64', 'I']]",
            topk.__doc__,
        )
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

    def test_numpy_topk_function(self):
        def mytopk(x, k):
            f = topk(x, k)
            return f

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

        f = jit_onnx(topk)
        res = f(x, k)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(y, res[0])
        self.assertEqualArray(z, res[1])

    def test_numpy_topk_function_indices(self):
        def mytopk(x, k):
            f = topk(x, k)
            return f[1]

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={"X": Float64[None], "K": Int64[1], (0, False): Int64[None]}
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    def test_numpy_topk_inline(self):
        f = topk_inline(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", topk.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", topk.__doc__)
        self.assertIn("k: TensorType['int64', (1,), 'I']", topk.__doc__)
        self.assertIn(
            ") -> TupleType[TensorType[numerics, 'T'], TensorType['int64', 'I']]",
            topk.__doc__,
        )
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

    def test_numpy_topk_function_inline(self):
        def mytopk(x, k):
            f = topk_inline(x, k)
            return f

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                "K": Int64[1],
                (0, False): Float64[None],
                (1, False): Int64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        y = np.array([[7, 6], [5, -6]], dtype=np.float64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

        f = jit_onnx(topk)
        res = f(x, k)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(y, res[0])
        self.assertEqualArray(z, res[1])

    def test_numpy_topk_function_indices_inline(self):
        def mytopk(x, k):
            f = topk_inline(x, k)
            return f[1]

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={"X": Float64[None], "K": Int64[1], (0, False): Int64[None]}
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        k = np.array([2], dtype=np.int64)
        z = np.array([[2, 1], [0, 1]], dtype=np.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x, "K": k})
        self.assertEqual(len(got), 1)
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    def test_numpy_min_max(self):
        def myf(x):
            f = _min_max(x)
            return f

        f = myf(Input("X"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                (0, False): Float64[None],
                (1, False): Float64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        z1 = np.array([[-7]], dtype=np.float64)
        z2 = np.array([[7]], dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(z1, got[0])
        self.assertEqualArray(z2, got[1])

        f = jit_onnx(myf)
        res = f(x)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(z1, res[0])
        self.assertEqualArray(z2, res[1])

    def test_numpy_min_max_inline(self):
        def myf(x):
            f = _min_max_inline(x)
            return f

        f = myf(Input("X"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={
                "X": Float64[None],
                (0, False): Float64[None],
                (1, False): Float64[None],
            }
        )
        x = np.array([[-5, 6, 7], [5, -6, -7]], dtype=np.float64)
        z1 = np.array([[-7]], dtype=np.float64)
        z2 = np.array([[7]], dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(z1, got[0])
        self.assertEqualArray(z2, got[1])

        f = jit_onnx(myf)
        res = f(x)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(z1, res[0])
        self.assertEqualArray(z2, res[1])

    def test_eager_numpy(self):
        def impl(A):
            print("A")
            b = absolute(A)
            print("B")
            c = b - A
            print("C")
            return c

        with redirect_stdout(StringIO()):
            f = impl(Input("A"))
            onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x) - x
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        with redirect_stdout(StringIO()):
            res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        with redirect_stdout(StringIO()):
            res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

        e = eager_onnx(impl)

        # Float64
        s = StringIO()
        with redirect_stdout(s):
            res = e(x)
        text = s.getvalue()
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)
        self.assertStartsWith("A\nA\nB\nC\n", text)

        # Int64
        s = StringIO()
        with redirect_stdout(s):
            res = e(x.astype(np.int64))
        text = s.getvalue()
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqual("A\nB\nC\n", text)

    @unittest.skipIf(InferenceSession is None, reason="onnxruntime is needed.")
    def test_eager_ort(self):
        def impl(A):
            print("A")
            b = absolute(A)
            print("B")
            c = b - A + cst([1])
            print("C")
            return c

        with redirect_stdout(StringIO()):
            f = impl(Input("A"))
            onx = f.to_onnx(constraints={"A": Float64[None], (0, False): Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x) - x + 1
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl, BackendOrtTensor, target_opsets={"": 17}, ir_version=8)

        # Float64
        xort = OrtTensor.from_array(x)
        with redirect_stdout(StringIO()):
            res = f(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)

        # Int64
        ix = x.astype(np.int64)
        xiort = OrtTensor.from_array(ix)
        with redirect_stdout(StringIO()):
            res = f(xiort)
        self.assertEqualArray(z.astype(np.int64), res.numpy())
        self.assertEqual(res.numpy().dtype, np.int64)

        e = eager_onnx(impl, EagerOrtTensor, target_opsets={"": 17})

        # Float64
        s = StringIO()
        with redirect_stdout(s):
            res = e(xort)
        text = s.getvalue()
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, np.float64)
        self.assertEqual(tuple(res.shape()), z.shape)
        self.assertStartsWith("A\nA\nB\nC\n", text)

        # Int64
        s = StringIO()
        with redirect_stdout(s):
            res = e(xiort)
        text = s.getvalue()
        self.assertEqual(res.numpy().dtype, np.int64)
        self.assertEqual("A\nB\nC\n", text)
        self.assertEqualArray(z.astype(np.int64), res.numpy())
        self.assertEqual(ix.shape, tuple(res.shape()))

    def common_numpy_op(self, msg, fct, use_int=False):
        if use_int:
            dtype = np.int64
            otype = Float64
        else:
            dtype = np.float64  # type: ignore[assignment]
            otype = Int64
        with self.subTest(msg=msg, op=fct):
            f = copy(fct(copy(Input("A")), Input("B")))
            self.assertIsInstance(f, Var)
            onx = f.to_onnx(constraints={"A": otype[None], "B": otype[None]})
            x = np.array([-5, 6], dtype=dtype)
            y = np.array([15, -16], dtype=dtype)
            z = fct(x, y)
            ref = ReferenceEvaluator(onx)
            got = ref.run(None, {"A": x, "B": y})
            try:
                self.assertEqualArray(z, got[0])
            except AssertionError as e:
                with open("debug_bin.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                raise AssertionError(f"Discrepancies with\n{onx}") from e

    def test_numpy_op_op(self):
        self.common_numpy_op("+", lambda x, y: x + y)
        self.common_numpy_op("-", lambda x, y: x - y)
        self.common_numpy_op("*", lambda x, y: x * y)
        self.common_numpy_op("/", lambda x, y: x / y)
        self.common_numpy_op("@", lambda x, y: x @ y)
        self.common_numpy_op("%", lambda x, y: x % y, True)

    def test_numpy_op_cmp(self):
        self.common_numpy_op("<", lambda x, y: x < y)
        self.common_numpy_op("<=", lambda x, y: x <= y)
        self.common_numpy_op(">", lambda x, y: x > y)
        self.common_numpy_op(">=", lambda x, y: x >= y)
        self.common_numpy_op("==", lambda x, y: x == y)
        self.common_numpy_op("!=", lambda x, y: x != y)

    def test_numpy_op_neg(self):
        self.common_numpy_op("-", lambda x, y: (-x) != y)

    def test_numpy_op_shift(self):
        self.common_numpy_op("<<", lambda x, y: x << y, True)
        self.common_numpy_op(">>", lambda x, y: x >> y, True)

    def test_numpy_op_bit(self):
        self.common_numpy_op("&", lambda x, y: x & y, True)
        self.common_numpy_op("|", lambda x, y: x | y, True)
        self.common_numpy_op("|", lambda x, y: x ^ y, True)
        self.common_numpy_op("~", lambda x, y: (~x) | y, True)

    def common_numpy_op_right(self, msg, fct, use_int=False):
        if use_int:
            dtype = np.int64
            otype = Float64
        else:
            dtype = np.float64  # type: ignore[assignment]
            otype = Int64
        if msg == "@":
            ccc = np.array([[1, 1]], dtype=dtype).T
            x = np.array([[-5, 6]], dtype=dtype)
        else:
            ccc = 1  # type: ignore[assignment]
            x = np.array([-5, 6], dtype=dtype)
        with self.subTest(msg=msg, op=fct):
            z = fct(ccc, x)
            f = copy(fct(ccc, copy(Input("A"))))
            self.assertIsInstance(f, Var)
            onx = f.to_onnx(constraints={"A": otype[None]})
            ref = ReferenceEvaluator(onx)
            got = ref.run(None, {"A": x})
            try:
                self.assertEqualArray(z, got[0])
            except AssertionError as e:
                with open("debug_bin.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                raise AssertionError(f"Discrepancies with\n{onx}") from e

    def test_numpy_op_op_right(self):
        self.common_numpy_op_right("+", lambda x, y: x + y)
        self.common_numpy_op_right("-", lambda x, y: x - y)
        self.common_numpy_op_right("*", lambda x, y: x * y)
        self.common_numpy_op_right("/", lambda x, y: x / y)
        self.common_numpy_op_right("%", lambda x, y: x % y, True)
        self.common_numpy_op_right("<", lambda x, y: x < y)
        self.common_numpy_op_right("<=", lambda x, y: x <= y)
        self.common_numpy_op_right(">", lambda x, y: x > y)
        self.common_numpy_op_right(">=", lambda x, y: x >= y)
        self.common_numpy_op_right("==", lambda x, y: x == y)
        self.common_numpy_op_right("!=", lambda x, y: x != y)
        self.common_numpy_op_right("&", lambda x, y: x & y, True)
        self.common_numpy_op_right("|", lambda x, y: x | y, True)
        self.common_numpy_op_right("|", lambda x, y: x ^ y, True)
        self.common_numpy_op_right("~", lambda x, y: (~x) | y, True)

    def test_shape(self):
        f = absolute_inline(Input("A").reshape(copy_inline(Input("A")).shape))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x.reshape(x.shape))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_shape_t(self):
        f = absolute_inline(Input("A").reshape(copy_inline(Input("A")).T.shape))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.reshape(x.T.shape))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_astype(self):
        f = absolute_inline(copy_inline(Input("A")).astype(np.float32))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.astype(np.float32))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_astype_int(self):
        f = absolute_inline(copy_inline(Input("A")).astype(1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.astype(np.float32))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_sum(self):
        f = absolute_inline(copy_inline(Input("A")).sum())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.sum())
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_copy(self):
        f = absolute_inline(Input("A").copy())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_flatten(self):
        f = absolute_inline(Input("A").flatten())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6], [-5, 6]], dtype=np.float64)
        z = np.abs(x.flatten())
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_sum_axis(self):
        f = absolute_inline(copy_inline(Input("A")).sum(axis=1, keepdims=1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(x.sum(axis=1, keepdims=1))  # type: ignore[call-overload]
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_numpy_op_bin_reduce(self):
        self.common_numpy_op(
            "and", lambda x, y: (x.sum() == y.sum()) & (((-x).sum()) == y.sum())
        )
        self.common_numpy_op(
            "or", lambda x, y: (x.sum() == y.sum()) | (((-x).sum()) == y.sum())
        )
        self.common_numpy_op(
            "xor", lambda x, y: (x.sum() == y.sum()) ^ (((-x).sum()) == y.sum())
        )

    def common_test_inline(self, fonx, fnp, tcst=0):
        f = fonx(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, 0.2], dtype=np.float64)
        x = x + tcst
        y = fnp(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0], atol=1e-10)

    def common_test_inline_bin(self, fonx, fnp, tcst=0):
        f = fonx(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Float64[None], 1: Float64[None], (0, False): Float64[None]}
        )
        x = np.array([[0.1, 0.2], [0.6, 10]], dtype=np.float64)
        y = np.array([[-1, 2], [-0.7, 0.1]], dtype=np.float64)
        x = x + tcst
        z = fnp(x, y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0], atol=1e-10)

    def test_arccos(self):
        self.common_test_inline(arccos_inline, np.arccos)

    def test_arccosh(self):
        self.common_test_inline(arccosh_inline, np.arccosh, tcst=1)

    def test_arcsin(self):
        self.common_test_inline(arcsin_inline, np.arcsin)

    def test_arcsinh(self):
        self.common_test_inline(arcsinh_inline, np.arcsinh)

    def test_arctan(self):
        self.common_test_inline(arctan_inline, np.arctan)

    def test_arctanh(self):
        self.common_test_inline(arctanh_inline, np.arctanh)

    def test_ceil(self):
        self.common_test_inline(ceil_inline, np.ceil)

    def test_clip(self):
        # 1
        f = clip_inline(Input("A"), cst(0), cst(1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, 0, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

        # 2
        f = clip_inline(Input("A"), cst(0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, 0, None)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_clip_int(self):
        f = clip_inline(Input("A"), 0, 1)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, 0, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_clip_none(self):
        f = clip_inline(Input("A"), None, cst(0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None], (0, False): Float64[None]})
        x = np.array([0.1, -0.2, 1.5], dtype=np.float64)
        y = np.clip(x, None, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

    def test_arange_inline(self):
        # arange(5)
        f = arange_inline(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[None], (0, False): Int64[None]})
        x = np.array(5, dtype=np.int64)
        y = np.arange(x)  # type: ignore[call-overload]
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(y, got[0])

        # arange(1, 5)
        f = arange_inline(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[1], 1: Int64[1], (0, False): Int64[None]})
        x1 = np.array(1, dtype=np.int64)
        x2 = np.array(5, dtype=np.int64)
        y = np.arange(x1, x2)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x1, "B": x2})
        self.assertEqualArray(y, got[0])

        # arange(1, 5, 2)
        f = arange_inline(Input("A"), Input("B"), Input("C"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Int64[1], 1: Int64[1], 2: Int64[1], (0, False): Int64[None]}
        )
        x1 = np.array(1, dtype=np.int64)
        x2 = np.array(5, dtype=np.int64)
        x3 = np.array(2, dtype=np.int64)
        y = np.arange(x1, x2, x3)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x1, "B": x2, "C": x3})
        self.assertEqualArray(y, got[0])

    def test_arange_inline_dtype(self):
        # arange(1, 5, 2), dtype
        f = arange_inline(Input("A"), Input("B"), Input("C"), dtype=np.float64)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Int64[1], 1: Int64[1], 2: Int64[1], (0, False): Int64[None]}
        )
        x1 = np.array(1, dtype=np.int64)
        x2 = np.array(5, dtype=np.int64)
        x3 = np.array(2, dtype=np.int64)
        y = np.arange(x1, x2, x3, dtype=np.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x1, "B": x2, "C": x3})
        self.assertEqual(y.dtype, got[0].dtype)
        self.assertEqualArray(y, got[0])

    def test_cos(self):
        self.common_test_inline(cos_inline, np.cos)

    def test_cosh(self):
        self.common_test_inline(cosh_inline, np.cosh)

    def test_compress_float32(self):
        x = np.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=np.float32)
        cond = np.array([False, True])

        axes = [0, 1, None]
        for axis in axes:
            with self.subTest(axis=axis):
                z = np.compress(cond, x, axis=axis)
                f = compress_inline(Input("A"), Input("B"), axis=axis)
                onx = f.to_onnx(constraints={"A": Bool[None], "B": Float32[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": cond, "B": x})
                self.assertEqualArray(z, got[0])

    def test_cumsum(self):
        x = np.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=np.float32)
        axis = np.array([1])

        z = np.cumsum(x, axis[0])
        f = cumsum_inline(Input("A"), Input("B"))
        onx = f.to_onnx(constraints={"A": Float32[None], "B": Int64[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": axis})
        self.assertEqualArray(z, got[0])

    def test_cumsum_no_axis(self):
        x = np.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=np.float32)

        z = np.cumsum(x)
        f = cumsum_inline(Input("A"))
        onx = f.to_onnx(constraints={"A": Float32[None]})
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_det(self):
        self.common_test_inline(det_inline, np.linalg.det, tcst=np.identity(2))

    def test_dot(self):
        self.common_test_inline_bin(dot_inline, np.dot)

    def test_einsum(self):
        equation = "ij,jk->ik"
        self.common_test_inline_bin(
            lambda x, y: einsum_inline(x, y, equation=equation),
            lambda x, y: np.einsum(equation, x, y),
        )

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_erf(self):
        self.common_test_inline(erf_inline, scipy.special.erf)

    def test_exp(self):
        self.common_test_inline(exp_inline, np.exp)

    def test_expand_dims(self):
        f = expand_dims_inline(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(
            constraints={0: Float64[None], 1: Int64[None], (0, False): Float64[None]}
        )
        x = np.array([[0.1, 0.2], [0.6, 10]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.int64)
        z = np.expand_dims(x, tuple(y))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0])

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_expit(self):
        self.common_test_inline(expit_inline, scipy.special.expit)

    def test_floor(self):
        self.common_test_inline(floor_inline, np.floor)

    def test_hstack(self):
        f = hstack_inline(Input("A"), Input("B"))
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2], [10, 20]], dtype=np.float64)
        z = np.hstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_identity(self):
        f = identity_inline(2, dtype=np.float64)
        onx = f.to_onnx(constraints={(0, False): Float64[None]})
        z = np.identity(2)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {})
        self.assertEqualArray(z, got[0])

    def test_isnan(self):
        self.common_test_inline(isnan_inline, np.isnan)

    def test_log(self):
        self.common_test_inline(log_inline, np.log)

    def test_log1p(self):
        self.common_test_inline(log1p_inline, np.log1p)

    def test_matmul(self):
        self.common_test_inline_bin(matmul_inline, np.matmul)

    def test_pad_1(self):
        x = np.random.randn(1, 3, 4, 5).astype(np.float64)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
        value = np.array(1.2, dtype=np.float64)

        for mode in ["constant", "reflect", "edge", "wrap"]:
            with self.subTest(mode=mode):
                z = pad_impl(x, pads, mode, 1.2)
                f = pad_inline(
                    copy_inline(Input("A")), cst(pads), cst(value), mode=mode
                )
                self.assertIsInstance(f, Var)
                onx = f.to_onnx(constraints={"A": Float64[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x})
                self.assertEqualArray(z, got[0])

    def test_pad_2(self):
        x = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
        value = np.array(1.2, dtype=np.float64)
        axes = np.array([1, 2, 3, 4], dtype=np.int64)

        for mode in ["constant", "reflect", "edge"]:
            with self.subTest(mode=mode):
                z = pad_impl(x, pads, mode, value, axes)
                f = pad_inline(
                    copy_inline(Input("A")), cst(pads), cst(value), cst(axes), mode=mode
                )
                self.assertIsInstance(f, Var)
                onx = f.to_onnx(constraints={"A": Float64[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x})
                self.assertEqualArray(z, got[0])

    def test_pad_3(self):
        x = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
        axes = np.array([1, 2, 3, 4], dtype=np.int64)

        for mode in ["constant", "reflect", "edge"]:
            with self.subTest(mode=mode):
                z = pad_impl(x, pads, mode, 0, axes)
                f = pad_inline(
                    copy_inline(Input("A")), cst(pads), None, cst(axes), mode=mode
                )
                self.assertIsInstance(f, Var)
                onx = f.to_onnx(constraints={"A": Float64[None]})
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x})
                self.assertEqualArray(z, got[0])

    def common_reduce(self, fct):
        f = absolute_inline(fct(copy_inline(Input("A"))))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.abs(fct(x))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_reduce_sum(self):
        self.common_reduce(lambda x: x.sum())

    def test_reduce_mean(self):
        self.common_reduce(lambda x: x.mean())

    def test_reduce_min(self):
        self.common_reduce(lambda x: x.min())

    def test_reduce_max(self):
        self.common_reduce(lambda x: x.max())

    def test_reduce_prod(self):
        self.common_reduce(lambda x: x.prod())

    def test_relu(self):
        self.common_test_inline(relu_inline, lambda x: np.where(x > 0, x, 0))

    def test_reciprocal(self):
        self.common_test_inline(reciprocal_inline, np.reciprocal)

    def test_round(self):
        self.common_test_inline(round_inline, np.round)

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_sigmoid(self):
        self.common_test_inline(sigmoid_inline, scipy.special.expit)

    def test_sign(self):
        self.common_test_inline(sign_inline, np.sign)

    def test_sin(self):
        self.common_test_inline(sin_inline, np.sin)

    def test_sinh(self):
        self.common_test_inline(sinh_inline, np.sinh)

    def test_sqrt(self):
        self.common_test_inline(sqrt_inline, np.sqrt)

    def test_squeeze(self):
        axis = np.array([1], dtype=np.int64)
        f = squeeze_inline(copy_inline(Input("A")), cst(axis))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64).T
        z = np.squeeze(x, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_squeeze_noaxis(self):
        f = squeeze_inline(copy_inline(Input("A")))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64)
        z = np.squeeze(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_tan(self):
        self.common_test_inline(tan_inline, np.tan)

    def test_tanh(self):
        self.common_test_inline(tanh_inline, np.tanh)

    def test_transpose(self):
        f = transpose_inline(copy_inline(Input("A")), perm=(1, 0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64).T
        z = np.transpose(x, (1, 0))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_unsqueeze(self):
        axis = np.array([1], dtype=np.int64)
        f = unsqueeze_inline(copy_inline(Input("A")), cst(axis))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([[-5, 6]], dtype=np.float64).T
        z = np.expand_dims(x, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_vstack(self):
        f = vstack_inline(Input("A"), Input("B"))
        onx = f.to_onnx(
            constraints={
                "A": Float64[None],
                "B": Float64[None],
                (0, False): Float64[None],
            }
        )
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2], [10, 20]], dtype=np.float64)
        z = np.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {"A": x1, "B": x2}
        got = ref.run(None, feeds)
        self.assertEqualArray(z, got[0])

    def test_where(self):
        zero = np.array([0], dtype=np.float64)
        f = where_inline(copy_inline(Input("A")) >= cst(zero), Input("A"), cst(zero))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={"A": Float64[None]})
        x = np.array([-5, 6], dtype=np.float64).T
        z = np.where(x >= 0, x, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_types(self):
        one = np.array([1], dtype=np.float64)

        def impl(x):
            return absolute_inline(copy_inline(x) + cst(one))

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_numpy_operator_types_array(self):
        one = np.array([1], dtype=np.float64)

        def impl(x):
            return absolute_inline(copy_inline(x) + one)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_numpy_operator_types_int(self):
        one = 1

        def impl(x):
            return absolute_inline(copy_inline(x) + one)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_numpy_operator_types_int_right(self):
        one = 1

        def impl(x):
            return absolute_inline(one + copy_inline(x))

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.array([-5, 6], dtype=np.float64)
        z = np.abs(x + 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def common_test_indices_int_tuple_slice(self, indices):
        def impl(x):
            return copy_inline(x)[indices]

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(63).reshape((9, 7)).astype(dtype=np.float64)
        z = x[indices]
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_indices_int_tuple_slice(self):
        self.common_test_indices_int_tuple_slice(1)
        self.common_test_indices_int_tuple_slice((1, 2))
        self.common_test_indices_int_tuple_slice(slice(0, 2))
        self.common_test_indices_int_tuple_slice((slice(0, 2), slice(4, 6)))
        self.common_test_indices_int_tuple_slice((slice(0, 2), 5))
        self.common_test_indices_int_tuple_slice((5, slice(0, 2)))
        self.common_test_indices_int_tuple_slice((5, slice(0, 7, 2)))

    def test_filter(self):
        def impl(x):
            y = copy_inline(x)
            ind = (y == 2) | (y == 8)
            return y[ind]

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(63).reshape((9, 7)).astype(dtype=np.float64)
        z = x[(x == 2) | (x == 8)]
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_int(self):
        def impl(x):
            y = copy_inline(x)
            return y.set[5](-6)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[5] = -6
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_slice(self):
        def impl(x):
            y = copy_inline(x)
            return y.set[5:8](np.array([-6, -7, -8]))

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[5:8] = np.array([-6, -7, -8])
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_where(self):
        def impl(x):
            y = copy_inline(x)
            return y.set[x == 5](-7)

        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[x == 5] = -7
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_where_set(self):
        def impl(x):
            y = copy_inline(x)
            y[x == 5] = -7
            return y()

        self.assertEmpty(Input("A").current_var_)
        i = Input("A")
        self.assertEqual(id(i), id(i.self_var))
        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[x == 5] = -7
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    def test_set_where_set_2(self):
        def impl(x):
            y = copy_inline(x)
            y[x == 5] = -7
            return y

        self.assertEmpty(Input("A").current_var_)
        i = Input("A")
        self.assertEqual(id(i), id(i.self_var))
        onx = impl(Input("A")).to_onnx(
            constraints={"A": Float64[None], (0, False): Float64[None]}
        )
        x = np.arange(10).astype(dtype=np.float64)
        z = x.copy()
        z[x == 5] = -7
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, np.float64)

        # Int64
        res = f(x.astype(np.int64))
        self.assertEqualArray(z.astype(np.int64), res)
        self.assertEqual(res.dtype, np.int64)

    @unittest.skipIf(InferenceSession is None, reason="onnxruntime is not available")
    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_cdist_com_microsoft(self):
        from scipy.spatial.distance import cdist as scipy_cdist

        metric = "euclidean"

        def impl(xa, xb):
            return cdist_inline(xa, xb, metric=metric)

        target_opsets = {"": 18, "com.microsoft": 1}
        onx = impl(Input("A"), Input("B")).to_onnx(
            constraints={
                "A": Float32[None],
                "B": Float32[None],
                (0, False): Float32[None],
            },
            target_opsets=target_opsets,
        )
        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        y = (np.arange(14).reshape((7, 2)) * 10).astype(dtype=np.float32)
        z = scipy_cdist(x, y, metric=metric).astype(np.float32)
        ref = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, {"A": x, "B": y})
        self.assertEqualArray(z, got[0], atol=1e-4)

        f = jit_onnx(impl, BackendOrtTensor, target_opsets=target_opsets)

        # float32
        xort = OrtTensor.from_array(x)
        yort = OrtTensor.from_array(y)
        self.assertEqualArray(x, xort.numpy())
        self.assertEqualArray(y, yort.numpy())
        res = f(xort, yort)
        self.assertEqual(res.numpy().dtype, np.float32)
        self.assertEqualArray(z, res.numpy(), atol=1e-4)

        # float64
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        xort = OrtTensor.from_array(x)
        yort = OrtTensor.from_array(y)
        self.assertEqualArray(x.astype(np.float64), xort.numpy())
        self.assertEqualArray(y.astype(np.float64), yort.numpy())
        res = f(xort, yort)
        self.assertEqual(res.numpy().dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res.numpy(), atol=1e-5)

        pieces = str(onx).split('s: "euclidean"')
        if len(pieces) > 2:
            raise AssertionError(f"Function is not using argument:\n{onx}")

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_cdist(self):
        from scipy.spatial.distance import cdist as scipy_cdist

        for metric in ["euclidean", "sqeuclidean"]:
            with self.subTest(metric=metric):

                def impl(xa, xb, metric=metric):
                    return cdist_inline(xa, xb, metric=metric)

                onx = impl(Input("A"), Input("B"), metric=metric).to_onnx(
                    constraints={
                        "A": Float64[None],
                        "B": Float64[None],
                        (0, False): Float64[None],
                    }
                )
                x = np.arange(10).reshape((5, 2)).astype(dtype=np.float64)
                y = np.arange(14).reshape((7, 2)).astype(dtype=np.float64) * 10
                z = scipy_cdist(x, y, metric=metric)
                ref = ReferenceEvaluator(onx)
                got = ref.run(None, {"A": x, "B": y})
                self.assertEqualArray(z, got[0])

                f = jit_onnx(impl)

                # Float64
                res = f(x, y)
                self.assertEqualArray(z, res)
                self.assertEqual(res.dtype, np.float64)

                # Float32
                res = f(x.astype(np.float32), y.astype(np.float32))
                self.assertEqualArray(z.astype(np.float32), res)
                self.assertEqual(res.dtype, np.float32)

    def test_onnx_in_var_node_proto(self):
        def impl(xa, xb):
            return xa + xb

        onx_base = impl(Input("A"), Input("B")).to_onnx(
            constraints={
                "A": Float32[None],
                "B": Float32[None],
                (0, False): Float32[None],
            }
        )
        self.assertIn("Add", str(onx_base))

        def impl2(x):
            return compute_inline(
                x,
                cst(np.array([5, 6], dtype=np.float32)).astype(x),
                proto=onx_base.graph.node[0],
            )

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )
        self.assertIn("Add", str(onx))

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = x + np.array([5, 6], dtype=np.float32)
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    @unittest.skipIf(scipy is None, reason="scipy is not installed.")
    def test_onnx_in_var_model_proto(self):
        from scipy.spatial.distance import cdist as scipy_cdist

        metric = "sqeuclidean"

        def impl(xa, xb):
            return cdist_inline(xa, xb, metric=metric)

        onx_base = impl(Input("xa"), Input("xb")).to_onnx(
            constraints={
                "xa": Float32[None],
                "xb": Float32[None],
                (0, False): Float32[None],
            }
        )
        self.assertNotIn("ai.onnx.ml", str(onx_base))

        def impl2(x):
            return compute_inline(
                x,
                cst(np.arange(4).reshape((2, 2)).astype(np.float32)).astype(x),
                proto=onx_base,
                name="mycdist",
            )

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = scipy_cdist(
            x, np.arange(4).reshape((2, 2)).astype(np.float32), metric=metric
        ).astype(np.float32)
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    def test_onnx_in_var_function_proto(self):
        def impl(xa, xb):
            return (xa - xb) ** 2

        onx_base = impl(Input("xa"), Input("xb")).to_onnx(
            constraints={
                "xa": Float32[None],
                "xb": Float32[None],
                (0, False): Float32[None],
            },
            as_function=True,
            name="diff_square",
            domain="local_f",
        )
        self.assertIsInstance(onx_base, FunctionProto)
        self.assertNotIn("ai.onnx.ml", str(onx_base))

        def impl2(x):
            return compute_inline(
                x,
                cst(np.arange(2).reshape((1, 2)).astype(np.float32)).astype(x),
                proto=onx_base,
                name="mycdist",
            )

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = ((x - np.arange(2).reshape((1, 2))) ** 2).astype(np.float32)
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    def test_onnx_in_var_model_proto_if(self):
        def _make_model():
            X = make_tensor_value_info(
                "X", TensorProto.FLOAT, ["N"]
            )  # pylint: disable=E1101
            Z = make_tensor_value_info(
                "Z", TensorProto.UNDEFINED, ["N"]
            )  # pylint: disable=E1101
            one = make_tensor_value_info(
                "one", TensorProto.FLOAT, ["N"]
            )  # pylint: disable=E1101

            graph1 = make_graph([], "then", [], [X])
            graph2 = make_graph([], "else", [], [one])

            graph_def = make_graph(
                [
                    make_node("ReduceSum", ["X"], ["Xred"]),
                    make_node("Constant", [], ["one"], value_floats=[1.0]),
                    make_node("CastLike", ["one", "Xred"], ["one_c"]),
                    make_node("Greater", ["Xred", "one_c"], ["cond"]),
                    make_node(
                        "If", ["cond"], ["Z_c"], then_branch=graph1, else_branch=graph2
                    ),
                    make_node("CastLike", ["Z_c", "X"], ["Z"]),
                ],
                "test",
                [X],
                [Z],
            )

            model_def = make_model(
                graph_def,
                producer_name="npx",
                ir_version=7,
                producer_version="0.1",
                opset_imports=[make_operatorsetid("", 15)],
            )
            return model_def

        def impl2(x):
            return compute_inline(x, proto=_make_model(), name="myif")

        onx = impl2(Input("A")).to_onnx(
            constraints={"A": Float32[None], (0, False): Float32[None]}
        )

        x = np.arange(10).reshape((5, 2)).astype(dtype=np.float32)
        z = x
        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"A": x})
        self.assertEqualArray(z, got[0], atol=1e-5)

        f = jit_onnx(impl2)

        # float32
        res = f(x)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqualArray(z, res, atol=1e-4)

        # float64
        x = x.astype(np.float64)
        res = f(x)
        self.assertEqual(res.dtype, np.float64)
        self.assertEqualArray(z.astype(np.float64), res)

    def test_kmeans(self):
        def compute_labels(X, centers):
            dist = cdist_inline(X, centers)
            return argmin_inline(dist, axis=1)

        onx = compute_labels(Input("X"), Input("centers")).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
            }
        )

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)

        f = jit_onnx(compute_labels)

        # float64
        res = f(x, centers)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)

    def test_kmeans_distance(self):
        def compute_labels(X, centers):
            dist = cdist_inline(X, centers)
            labels = argmin_inline(dist, axis=1)
            return make_tuple(labels, dist)

        onx = compute_labels(Input("X"), Input("centers")).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)
        self.assertEqual(got[1].dtype, np.float64)

        f = jit_onnx(compute_labels)

        # float64
        res, dist = f(x, centers)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)
        self.assertEqualArray(got[1], dist)

    def test_kmeans_distance_calls(self):
        def build_distance(X, centers, use_sqrt=False):
            dist = cdist_inline(X, centers, metric="sqeuclidean")
            if use_sqrt:
                return sqrt_inline(dist)
            return dist

        def compute_labels(X, centers):
            dist = build_distance(X, centers, True)
            labels = argmin_inline(dist, axis=1)
            return make_tuple(labels, dist)

        onx = compute_labels(Input("X"), Input("centers")).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )
        self.assertIn('"Sqrt"', str(onx))

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)
        self.assertEqual(got[1].dtype, np.float64)

        f = jit_onnx(compute_labels)
        self.assertEqual(len(f.onxs), 0)
        self.assertEqual(f.n_versions, 0)

        # float64
        res, dist = f(x, centers)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)
        self.assertEqualArray(got[1], dist)
        self.assertEqual(f.n_versions, 1)
        self.assertEqual(len(f.available_versions), 1)
        self.assertEqual(f.available_versions, [((np.float64, 2), (np.float64, 2))])
        key = ((np.dtype("float64"), 2), (np.dtype("float64"), 2))
        onx = f.get_onnx(key)
        self.assertIsInstance(onx, ModelProto)
        self.assertRaise(lambda: f.get_onnx(2), ValueError)
        onx = f.get_onnx()
        self.assertIsInstance(onx, ModelProto)

    def test_kmeans_distance_calls_args(self):
        def build_distance(X, centers, use_sqrt=False):
            dist = cdist_inline(X, centers, metric="sqeuclidean")
            if use_sqrt:
                return sqrt_inline(dist)
            return dist

        def compute_labels(X, centers, use_sqrt=False):
            dist = build_distance(X, centers, use_sqrt)
            labels = argmin_inline(dist, axis=1)
            return make_tuple(labels, dist)

        onx = compute_labels(Input("X"), Input("centers"), use_sqrt=False).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )
        self.assertNotIn('"Sqrt"', str(onx))

        onx = compute_labels(Input("X"), Input("centers"), use_sqrt=True).to_onnx(
            constraints={
                "X": Float64[None],
                "centers": Float64[None],
                (0, False): Int64[None],
                (1, False): Float64[None],
            }
        )
        self.assertIn('"Sqrt"', str(onx))

        x = np.random.randn(100, 2)
        centers = np.random.randn(2, 2)

        ref = ReferenceEvaluator(onx.SerializeToString())
        got = ref.run(None, {"X": x, "centers": centers})
        self.assertEqual(got[0].dtype, np.int64)
        if DEFAULT_OPSET > 18:
            self.assertEqual(got[0].min(), 0)
            self.assertEqual(got[0].max(), 1)
        self.assertEqual(got[1].dtype, np.float64)

        f = jit_onnx(compute_labels)
        self.assertEqual(len(f.onxs), 0)
        self.assertEqual(f.n_versions, 0)

        # float64
        res, dist = f(x, centers, use_sqrt=True)
        self.assertEqual(res.dtype, np.int64)
        self.assertEqualArray(got[0], res)
        self.assertEqualArray(got[1], dist)
        self.assertEqual(f.n_versions, 1)
        self.assertEqual(len(f.available_versions), 1)
        key = ((np.dtype("float64"), 2), (np.dtype("float64"), 2), "use_sqrt", True)
        self.assertEqual(f.available_versions, [key])
        onx = f.get_onnx(key)
        self.assertIsInstance(onx, ModelProto)
        self.assertRaise(lambda: f.get_onnx(2), ValueError)
        onx = f.get_onnx()
        self.assertIsInstance(onx, ModelProto)
        self.assertIn('"Sqrt"', str(onx))


if __name__ == "__main__":
    TestNpx().test_pad_2()
    unittest.main(verbosity=2)
