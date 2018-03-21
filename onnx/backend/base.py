from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import six

import onnx.checker


class DeviceType(object):
    CPU = 0
    CUDA = 1


class Device(object):
    '''
    Describes device type and device id
    syntax: device_type:device_id(optional)
    example: 'CPU', 'CUDA', 'CUDA:1'
    '''

    def __init__(self, device):
        options = device.split(':')
        self.type = getattr(DeviceType, options[0])
        self.device_id = 0
        if len(options) > 1:
            self.device_id = int(options[1])


def namedtupledict(typename, field_names, *args, **kwargs):
    field_names_map = {n: i for i, n in enumerate(field_names)}
    # Some output names are invalid python identifier, e.g. "0"
    kwargs.setdefault('rename', True)
    data = namedtuple(typename, field_names, *args, **kwargs)

    def getitem(self, key):
        if isinstance(key, six.string_types):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)
    data.__getitem__ = getitem
    return data


class BackendRep(object):
    def run(self, inputs, **kwargs):
        pass


class Backend(object):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        onnx.checker.check_model(model)

    @classmethod
    def run_model(cls, model, inputs, device='CPU', **kwargs):
        return cls.prepare(model, device, **kwargs).run(inputs)

    @classmethod
    def run_node(cls, node, inputs, device='CPU', outputs_info=None, **kwargs):
        '''Simple run one operator and return the results.
        Args:
            outputs_info: a list of tuples, which contains the element type and
            shape of each output. First element of the tuple is the dtype, and
            the second element is the shape. More use case can be found in
            https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py
        '''
        onnx.checker.check_node(node)

    @classmethod
    def supports_device(cls, device):
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return True
