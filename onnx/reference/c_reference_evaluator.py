# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415,R0902,R0912,R0913,R0914,R0915

from typing import Any, Dict, List, Optional, Union

from onnx import FunctionProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.c_ops import Conv
from onnx.reference.op_run import OpRun


class CReferenceEvaluator(ReferenceEvaluator):
    """
    This class replaces the python implementation by C implementation
    for a short list of operators quite slow in python (such as `Conv`).
    The class automatically replaces a python implementation
    by a C implementation if available.

    ::

        from onnx.reference import ReferenceEvaluator
        from from onnx.reference.c_ops import Conv
        ref = ReferenceEvaluator(..., new_ops=[Conv])
    """

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,  # type: ignore
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
    ):
        if new_ops is None:
            new_ops = [Conv]
        else:
            new_ops = new_ops.copy()
            new_ops.append(Conv)
        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
        )
