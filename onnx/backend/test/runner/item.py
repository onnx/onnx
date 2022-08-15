# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, List, Optional, Union

from onnx import ModelProto, NodeProto

# A container that hosts the test function and the associated
# test item (ModelProto)


class TestItem:
    def __init__(
        self,
        func: Callable[..., Any],
        proto: List[Optional[Union[ModelProto, NodeProto]]],
    ) -> None:
        self.func = func
        self.proto = proto
