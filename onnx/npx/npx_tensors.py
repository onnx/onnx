# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unnecessary-pass

from typing import Any, Callable, List, Tuple

import numpy as np

from onnx import ModelProto
from onnx.npx.npx_array_api import ArrayApi
from onnx.npx.npx_types import TensorType
from onnx.reference import ReferenceEvaluator


class JitTensor:
    """
    Defines a value for a specific jit mode
    """

    pass


class EagerTensor(ArrayApi):
    """
    Defines a value for a specific eager mode.
    An eager tensor must overwrite every call to a method listed in class
    :class:`ArrayApi`.
    """

    pass
