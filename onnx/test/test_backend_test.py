from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
import unittest
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import Device, DeviceType
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
import onnx.shape_inference
import onnx.version_converter
from typing import Optional, Text, Any, Tuple, Sequence
from onnx import NodeProto, ModelProto, TensorProto
import numpy  # type: ignore

# The following just executes the fake backend through the backend test
# infrastructure. Since we don't have full reference implementation of all ops
# in ONNX repo, it's impossible to produce the proper results. However, we can
# run 'checker' (that's what base Backend class does) to verify that all tests
# fed are actually well-formed ONNX models.
#
# If everything is fine, all the tests would be marked as "skipped".
#
# We don't enable report in this test because the report collection logic itself
# fails when models are mal-formed.


class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[onnx.backend.base.BackendRep]
        super(DummyBackend, cls).prepare(model, device, **kwargs)

        # test shape inference
        model = onnx.shape_inference.infer_shapes(model)
        value_infos = {vi.name: vi for vi in itertools.chain(model.graph.value_info, model.graph.output)}

        if do_enforce_test_coverage_whitelist(model):
            for node in model.graph.node:
                for i, output in enumerate(node.output):
                    if node.op_type == 'Dropout' and i != 0:
                        continue
                    assert output in value_infos
                    tt = value_infos[output].type.tensor_type
                    assert tt.elem_type != TensorProto.UNDEFINED
                    for dim in tt.shape.dim:
                        assert dim.WhichOneof('value') == 'dim_value'

        raise BackendIsNotSupposedToImplementIt(
            "This is the dummy backend test that doesn't verify the results but does run the checker")

    @classmethod
    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Any
                 ):  # type: (...) -> Optional[Tuple[Any, ...]]
        super(DummyBackend, cls).run_node(node, inputs, device=device, outputs_info=outputs_info)
        raise BackendIsNotSupposedToImplementIt(
            "This is the dummy backend test that doesn't verify the results but does run the checker")

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False


test_coverage_whitelist = set(
    ['bvlc_alexnet', 'densenet121', 'inception_v1', 'inception_v2',
     'resnet50', 'shufflenet', 'SingleRelu', 'squeezenet_old', 'vgg19', 'zfnet'])


def do_enforce_test_coverage_whitelist(model):  # type: (ModelProto) -> bool
    if model.graph.name not in test_coverage_whitelist:
        return False
    for node in model.graph.node:
        if node.op_type in set(['RNN', 'LSTM', 'GRU']):
            return False
    return True


backend_test = onnx.backend.test.BackendTest(DummyBackend, __name__)
if os.getenv('APPVEYOR'):
    backend_test.exclude(r'(test_vgg19|test_zfnet)')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
