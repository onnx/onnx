from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import os
import tarfile
import tempfile
import unittest

import numpy as np

import onnx
from onnx import helper
from .test_util import create_input, N
from six.moves.urllib.request import urlretrieve

L = 20
M = 10
S = 5
const2_np = np.random.randn(S, S)
const2_onnx = onnx.helper.make_tensor("const2",
                                      onnx.TensorProto.FLOAT,
                                      (S, S),
                                      const2_np.flatten().astype(float))

# TODO: These Numpy specs will be generally useful to backend implementations,
# so they should get moved out of here at some point
node_tests = [
  ("test_abs", N("Abs"), np.abs, [(S, S, S)]),
  ("test_add", N("Add"), np.add, [(S, S, S), (S, S, S)]),
  ("test_add_bcast", N("Add", broadcast=1), np.add, [(S, M), (M,)]),
  ("test_constant", N("Constant", value=const2_onnx), lambda: const2_np, []),
  # TODO: Are we actually supporting other dot modes?  In that case, some fancy
  # footwork is necessary...
  ("test_dot", N("Dot"), np.dot, [(S, M), (M, L)]),
  ("test_relu", N("Relu"), lambda x: np.clip(x, 0, np.inf), [(S, S, S)]),
  # TODO: Add all the other operators
]

graph_tests = [
    ('test_bvlc_alexnet', 'bvlc_alexnet'),
    ('test_densenet121', 'densenet121'),
    ('test_inception_v1', 'inception_v1'),
    ('test_inception_v2', 'inception_v2'),
    ('test_resnet50', 'resnet50'),
    ('test_shufflenet', 'shufflenet'),
    ('test_squeezenet', 'squeezenet'),
    ('test_vgg16', 'vgg16'),
    ('test_vgg19', 'vgg19'),
]

class BackendTest(object):
    def __init__(self, backend):
        class TestsContainer(unittest.TestCase):
            def setUp(self):
                np.random.seed(seed=0)

        self.backend = backend
        self.tests = TestsContainer

        for nt in node_tests:
            self._add_node_test(*nt)

        for gt in graph_tests:
            self._add_graph_test(*gt)

    def _prepare_model(self, model_name):
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', '~/.onnx'))
        models_dir = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(
                model_name)
            download_file = tempfile.NamedTemporaryFile(delete=True)
            print('Start downloading model {} from {}'.format(model_name, url))
            urlretrieve(url, download_file.name)
            print('Done')
            with tarfile.open(download_file.name) as t:
                t.extractall(models_dir)
        return model_dir

    def _add_test(self, name, test_func):
        # We don't prepend the 'test_' prefix to improve greppability
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        if hasattr(self.tests, name):
            raise ValueError('Duplicated test name: {}'.format(name))
        setattr(self.tests, name, test_func)

    def _add_graph_test(self, test_name, model_name):
        """
        Add A test for a single ONNX model against a reference implementation.
            test_name (string): Eventual name of the test.  Must be prefixed
                with 'test_'.
            model_name (string): The ONNX model's name
            inputs (list of ndarrays): inputs to the model
            outputs (list of ndarrays): outputs to the model
        """
        def run(test_self):
            model_dir = self._prepare_model(model_name)
            graph_pb_path = os.path.join(model_dir, 'graph.pb')
            graph = onnx.load(graph_pb_path)
            prepared_graph = self.backend.prepare(graph)

            for test_data_npz in glob.glob(os.path.join(model_dir, 'test_data_*.npz')):
                test_data = np.load(test_data_npz)
                inputs = list(test_data['inputs'])
                outputs = list(prepared_graph.run(inputs))
                ref_outputs = test_data['outputs']
                test_self.assertEqual(len(ref_outputs), len(outputs))
                for i in range(len(outputs)):
                    np.testing.assert_almost_equal(
                        ref_outputs[i],
                        outputs[i],
                        decimal=4)

        self._add_test(test_name, run)

    def _add_node_test(self, test_name, node_spec, ref, inputs):
        """
        Add A test for a single ONNX node against a reference implementation.

        Arguments:
            test_name (string): Eventual name of the test.  Must be prefixed
                with 'test_'.
            node (NodeSpec): The ONNX node's name and attributes to be tested;
                inputs and outputs will be inferred from other arguments of this
                spec.
            ref (lambda): A function from any number of Numpy ndarrays,
                to a single ndarray or tuple of ndarrays.
            inputs (tuple of ndarrays or size tuples): A specification of
                the input to the operator.
        """
        def run(test_self):
            # TODO: In some cases we should generate multiple random inputs
            # and test (ala Hypothesis)
            args = create_input(inputs)
            ref_outputs = ref(*args)
            if not isinstance(ref_outputs, tuple):
                ref_outputs = (ref_outputs,)
            input_names = ['input_{}'.format(i) for i in range(len(args))]
            output_names = ['output_{}'.format(i) for i in range(len(ref_outputs))]
            node_def = helper.make_node(
                node_spec.name,
                input_names,
                output_names,
                **node_spec.kwargs)
            outputs = self.backend.run_node(node_def, args)
            test_self.assertEqual(len(ref_outputs), len(outputs))
            for i in range(len(output_names)):
                np.testing.assert_almost_equal(
                    ref_outputs[i],
                    outputs[i],
                    decimal=4)

        self._add_test(test_name, run)
