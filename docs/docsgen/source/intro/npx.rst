.. _l-numpy-api-onnx:

==================
Numpy API for ONNX
==================

Many users have difficulties to write onnx graphs.
Many packages tries to symplify it either by implementing
their own api very close to onnx operators
(`sklearn-onnx <http://onnx.ai/sklearn-onnx/>`_,
`tf2onnx <https://github.com/onnx/tensorflow-onnx>`_,
`spox <https://spox.readthedocs.io/en/latest/>`_,
`onnx-script <https://github.com/microsoft/onnx-script>`_).
This contribution tries a different approach by implementing
a numpy API for ONNX. It does not cover everything numpy
or ONNX can do but it can easily be used to define
loss functions for example without knowing too much about ONNX.

.. note:: control flow

    The first version (onnx==1.15) does not support control flow yet (test and loops).
    There is no easy syntax for that yet and the main challenge is to deal with local context.

Overview
========

Example
+++++++

Let's define the L1 loss computed from two vectors:

.. exec_code::

    import numpy as np
    from onnx.npx import jit_onnx
    from onnx.npx import absolute

    # The function looks like a numpy function.
    def l1_loss(x, y):
        return absolute(x - y).sum()

    # The function needs to be converted into ONNX with function jit_onnx.
    # jitted_l1_loss is a wrapper. It intercepts all calls to l1_loss.
    # When it happens, it checks the input types and creates the
    # corresponding ONNX graph.
    jitted_l1_loss = jit_onnx(l1_loss)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    # First execution and conversion to ONNX.
    # The wrapper caches the created onnx graph.
    # It reuses it if the input types and the number of dimension are the same.
    # It creates a new one otherwise and keep the old one.
    res = jitted_l1_loss(x, y)
    print(res)

    # The ONNX graph can be accessed the following way.
    print(jitted_l1_loss.get_onnx())


We can also define a more complex loss by computing L1 loss on
the first column and L2 loss on the seconde one.

.. exec_code::

    import numpy as np
    from onnx.npx import jit_onnx
    from onnx.npx import absolute

    def l1_loss(x, y):
        return absolute(x - y).sum()

    def l2_loss(x, y):
        return ((x - y) ** 2).sum()

    def myloss(x, y):
        return l1_loss(x[:, 0], y[:, 0]) + l2_loss(x[:, 1], y[:, 1])

    jitted_myloss = jit_onnx(myloss)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    res = jitted_myloss(x, y)
    print(res)

onnxruntime
+++++++++++

The backend is the class :class:`ReferenceEvalutor` by default but it could
be replaced by onnxruntime. The backend is not implemented in onnx package
but is added to the following example. The current implementation
is available with class `OrtTensor
<https://github.com/onnx/onnx/tree/main/onnx/test/npx_test.py#L100>`_.

.. exec_code::

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

            def __init__(self, tensor_class: type, input_names: List[str], onx: ModelProto):
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
                        onx.graph.output[0].type.tensor_type.elem_type = onx.graph.input[
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
                res = (
                    self.ref._sess.run_with_ort_values(  # pylint: disable=protected-access
                        feeds, self.output_names, self.run_options
                    )
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
        def create_function(cls: Any, input_names: List[str], onx: ModelProto) -> Callable:
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


    import numpy as np
    from onnx.npx import jit_onnx
    from onnx.npx import absolute


    def l1_loss(x, y):
        return absolute(x - y).sum()

    def l2_loss(x, y):
        return ((x - y) ** 2).sum()

    def myloss(x, y):
        l1 = l1_loss(x[:, 0], y[:, 0])
        l2 = l2_loss(x[:, 1], y[:, 1])
        return l1 + l2 

    ort_myloss = jit_onnx(myloss, BackendOrtTensor, target_opsets={"": 17}, ir_version=8)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)

    xort = OrtTensor.from_array(x)
    yort = OrtTensor.from_array(y)

    res = ort_myloss(xort, yort)
    print(res.numpy())

This backend do not support numpy array but only the 
class OrtValue which represents a tensor in onnxruntime.
This value can be easily created from a numpy array and could
be placed on CPU or CUDA if it is available.

Eager mode
++++++++++

Eager mode is fully supported yet.
