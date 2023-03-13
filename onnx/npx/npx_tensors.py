# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unnecessary-pass

from typing import Any, Callable, List, Tuple

import numpy as np

from onnx import ModelProto
from onnx.npx.npx_types import TensorType
from onnx.reference import ReferenceEvaluator


class EagerTensor:
    """
    Defines a value for a specific eager mode.
    """

    pass


class JitTensor:
    """
    Defines a value for a specific jit mode
    """

    pass
