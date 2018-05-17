from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Callable, Any, Union, List, Optional
from onnx import NodeProto, ModelProto


# A container that hosts the test function and the associated
# test item (ModelProto)


class TestItem(object):
    def __init__(self, func, proto):  # type: (Callable[..., Any], List[Optional[Union[ModelProto, NodeProto]]]) -> None
        self.func = func
        self.proto = proto
