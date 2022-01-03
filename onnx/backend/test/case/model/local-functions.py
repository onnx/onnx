# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import Sequence

from onnx import helper, parser, checker, compose, version_converter, \
    ModelProto, GraphProto, ValueInfoProto, TensorProto, SparseTensorProto, \
    FunctionProto, NodeProto


class LocalFunctions(Base):

    @staticmethod
    def export():  # type: () -> None

        ops = [
            helper.make_opsetid("", 10),
            helper.make_opsetid("local", 10)
        ]

        g = GraphProto()
        g.name = 'g'
        g.input.extend([
            helper.make_tensor_value_info('x0', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('x1', TensorProto.FLOAT, [])
        ])
        g.output.extend([
            helper.make_tensor_value_info('y', TensorProto.FLOAT, []),
        ])
        g.node.extend([
            helper.make_node(
                'f_node', domain='local', inputs=['x0', 'x1'], outputs=['y'])
        ])

        m1 = helper.make_model(g, producer_name='test', opset_imports=ops)
        m1.functions.extend([
            helper.make_function(
                'local', 'f', ['x0', 'x1'], ['y'],
                [helper.make_node('Add', inputs=['x0', 'x1'], outputs=['y'])],
                opset_imports=ops
            )
        ])
        checker.check_model(m1)

        input_shape = [1, 3, 1]
        x0 = np.ones(input_shape, dtype=np.float32)
        x1 = np.ones(input_shape, dtype=np.float32)
        y = x0 + x1
        expect(m1, inputs=[x0, x1], outputs=[y], name="test_local_funcion_add")
