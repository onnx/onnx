from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def create_input(call_args):
    """
    Creates a tuple of Numpy ndarray from a template 'call_args'; every embedded
    tuple in call_args is converted into a random ndarray with the specified
    dimensions.
    """
    def map_arg(arg):
        if isinstance(arg, tuple):
            return np.random.randn(*arg).astype(np.float32)
        else:
            return arg

    return [map_arg(arg) for arg in call_args]


class NodeSpec(object):
    """
    Describes a onnx.NodeProto, but without inputs/outputs
    (which will be inferred).
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

N = NodeSpec
