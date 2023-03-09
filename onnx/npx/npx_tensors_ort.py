# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
from onnx import ModelProto, TensorProto
from onnx.defs import onnx_opset_version
from onnxruntime import InferenceSession, RunOptions
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue,
    OrtDevice as C_OrtDevice,
    OrtMemType,
)
from onnx.npx.npx_types import TensorType
from onnx.npx.npx_tensors import BackendTensor, EagerTensor


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
        return self._tensor.numpy()

    class Evaluator:
        """
        Wraps class :class:`onnxruntime.InferenceSession`
        to have a signature closer to python function.
        """

        def __init__(self, tensor_class: type, input_names: List[str], onx: ModelProto):
            try:
                self.ref = InferenceSession(onx.SerializeToString())
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
                    self.ref = InferenceSession(onx.SerializeToString())
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
                feeds[name] = inp.value
            res = self.ref._sess.run_with_ort_values(
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
        return self._tensor.shape()

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
        return TensorType[self.dtype]

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
        return (None,) + tuple(self.shape[1:])

    @property
    def tensor_type_dims(self) -> TensorType:
        """
        Returns the tensor type of this tensor.
        This property is used to define a key used to cache a jitted function.
        Same keys keys means same ONNX graph.
        Different keys usually means same ONNX graph but different
        input shapes.
        """
        return TensorType[self.dtype, self.dims]

    @classmethod
    def create_function(cls: Any, input_names: List[str], onx: ModelProto) -> Callable:
        """
        Creates a python function calling the onnx backend
        used by this class.

        :param onx: onnx model
        :return: python function
        """
        return cls.Evaluator(cls, input_names, onx)


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
