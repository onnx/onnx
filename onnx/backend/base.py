# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from typing import Text, Sequence, Any, Type, Tuple, NewType, Optional, Dict

import numpy  # type: ignore

import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import ModelProto, NodeProto, IR_VERSION


class DeviceType(object):
    _Type = NewType('_Type', int)
    CPU: _Type = _Type(0)
    CUDA: _Type = _Type(1)


class Device(object):
    '''
    Describes device type and device id
    syntax: device_type:device_id(optional)
    example: 'CPU', 'CUDA', 'CUDA:1'
    '''

    def __init__(self, device: Text) -> None:
        options = device.split(':')
        self.type = getattr(DeviceType, options[0])
        self.device_id = 0
        if len(options) > 1:
            self.device_id = int(options[1])


def namedtupledict(typename: Text, field_names: Sequence[Text], *args: Any, **kwargs: Any) -> Type[Tuple[Any, ...]]:
    field_names_map = {n: i for i, n in enumerate(field_names)}
    # Some output names are invalid python identifier, e.g. "0"
    kwargs.setdefault(str('rename'), True)
    data = namedtuple(typename, field_names, *args, **kwargs)  # type: ignore

    def getitem(self: Any, key: Any) -> Any:
        if isinstance(key, str):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)  # type: ignore
    setattr(data, "__getitem__", getitem)
    return data


class BackendRep(object):
    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        pass


class Backend(object):
    @classmethod
    def is_compatible(cls,
                      model: ModelProto,
                      device: Text = 'CPU',
                      **kwargs: Any
                      ) -> bool:
        # Return whether the model is compatible with the backend.
        return True

    @classmethod
    def prepare(cls,
                model: ModelProto,
                device: Text = 'CPU',
                **kwargs: Any
                ) -> Optional[BackendRep]:
        # TODO Remove Optional from return type
        onnx.checker.check_model(model)
        return None

    @classmethod
    def run_model(cls,
                  model: ModelProto,
                  inputs: Any,
                  device: Text = 'CPU',
                  **kwargs: Any
                  ) -> Tuple[Any, ...]:
        backend = cls.prepare(model, device, **kwargs)
        assert backend is not None
        return backend.run(inputs)

    @classmethod
    def run_node(cls,
                 node: NodeProto,
                 inputs: Any,
                 device: Text = 'CPU',
                 outputs_info: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]] = None,
                 **kwargs: Dict[Text, Any]
                 ) -> Optional[Tuple[Any, ...]]:
        '''Simple run one operator and return the results.
        Args:
            outputs_info: a list of tuples, which contains the element type and
            shape of each output. First element of the tuple is the dtype, and
            the second element is the shape. More use case can be found in
            https://github.com/onnx/onnx/blob/main/onnx/backend/test/runner/__init__.py
        '''
        # TODO Remove Optional from return type
        if 'opset_version' in kwargs:
            special_context = c_checker.CheckerContext()
            special_context.ir_version = IR_VERSION
            special_context.opset_imports = {'': kwargs['opset_version']}  # type: ignore
            onnx.checker.check_node(node, special_context)
        else:
            onnx.checker.check_node(node)
        return None

    @classmethod
    def supports_device(cls, device: Text) -> bool:
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return True
