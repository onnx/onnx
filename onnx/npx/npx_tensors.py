# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, List, Tuple
import numpy as np
from onnx import ModelProto
from onnx.reference import ReferenceEvaluator
from onnx.npx.npx_types import TensorType


class NumpyTensor:
    """
    Default backend based on
    :func:`onnx.reference.ReferenceEvaluator`.

    :param input_names: input names
    :param onx: onnx model
    """

    class Evaluator:
        """
        Wraps class :class:`onnx.reference.ReferenceEvaluator`
        to have a signature closer to python function.
        """

        def __init__(self, tensor_class: type, input_names: List[str], onx: ModelProto):
            self.ref = ReferenceEvaluator(onx)
            self.input_names = input_names
            self.tensor_class = tensor_class

        def run(self, *inputs: List["NumpyTensor"]) -> List["NumpyTensor"]:
            """
            Executes the function.

            :param inputs: function inputs
            :return: outputs
            """
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs but got " f"len(inputs)."
                )
            feeds = {}
            for name, inp in zip(self.input_names, inputs):
                feeds[name] = inp.value
            return list(map(self.tensor_class, self.ref.run(None, feeds)))

    def __init__(self, tensor: np.ndarray):
        if isinstance(tensor, np.int64):
            tensor = np.array(tensor, dtype=np.int64)
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"A numpy array is expected not {type(tensor)}.")
        self._tensor = tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        "Returns the shape of the tensor."
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        "Returns the element type of this tensor."
        return self._tensor.dtype

    @property
    def key(self) -> Any:
        "Unique key for a tensor of the same type."
        return (self.dtype, len(self.shape))

    @property
    def value(self) -> np.ndarray:
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
            return self.shape
        return (None,) + self.shape[1:]

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

    @classmethod
    def get_opsets(cls, opsets):
        """
        Updates the opsets for a given backend.
        This method should be overloaded.
        By default, it returns opsets.
        """
        return opsets

    @classmethod
    def get_ir_version(cls, ir_version):
        """
        Updates the IR version.
        This method should be overloaded.
        By default, it returns ir_version.
        """
        return ir_version


class BackendEagerTensor:
    """
    Defines a value for a specific backend or eager mode.
    """

    pass


class BackendTensor(BackendEagerTensor):
    """
    Defines a value for a specific backend.
    """

    pass


class EagerTensor(BackendEagerTensor):
    """
    Defines a value for a specific eager mode.
    """

    pass


class BackendNumpyTensor(NumpyTensor, BackendTensor):
    """
    Defines a value for a specific backend.
    """

    pass


class EagerNumpyTensor(NumpyTensor, EagerTensor):
    """
    Defines a value for a specific backend.
    """

    pass
