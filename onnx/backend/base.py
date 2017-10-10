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
        cls.prepare(model, device, **kwargs).run(inputs)

    @classmethod
    def run_node(cls, node, inputs, device='CPU'):
        onnx.checker.check_node(node)
