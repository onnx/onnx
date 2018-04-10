from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
import onnx.backend.base
import onnx.backend.test
from onnx.backend.base import Device, DeviceType
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt


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
    def prepare(cls, model, device='CPU', **kwargs):
        super(DummyBackend, cls).prepare(model, device, **kwargs)
        raise BackendIsNotSupposedToImplementIt(
            "This is the dummy backend test that doesn't verify the results but does run the checker")

    @classmethod
    def run_node(cls, node, inputs, device='CPU', outputs_info=None):
        super(DummyBackend, cls).run_node(node, inputs, device=device, outputs_info=outputs_info)

        # test shape inference
        inputs_info = [(x.dtype, x.shape) for x in inputs]
        input_value_infos = [helper.make_tensor_value_info(x, NP_TYPE_TO_TENSOR_TYPE[t], shape)
                              for x, (t, shape) in zip(node.input, inputs_info)]
        output_value_infos = [helper.make_tensor_value_info(x, NP_TYPE_TO_TENSOR_TYPE[t], shape)
                               for x, (t, shape) in zip(node.output, outputs_info)]
        if outputs_info:
            graph = helper.make_graph([node], "test", input_value_infos, [])
            orig_model = helper.make_model(graph, producer_name='onnx-test')
            inferred_model = onnx.shape_inference.infer_shapes(orig_model)

            # Allow shape inference to not return anything, but if it
            # does then check that it's correct
            if inferred_model.graph.value_info:
                assert(list(inferred_model.graph.value_info) == output_value_infos)

        raise BackendIsNotSupposedToImplementIt(
            "This is the dummy backend test that doesn't verify the results but does run the checker")

    @classmethod
    def supports_device(cls, device):
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False


backend_test = onnx.backend.test.BackendTest(DummyBackend, __name__)
if os.getenv('APPVEYOR'):
    backend_test.exclude(r'(test_vgg19|test_vgg16)')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
