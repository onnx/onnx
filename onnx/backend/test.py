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
from . import test_rnn
from six.moves.urllib.request import urlretrieve

L = 20
M = 10
S = 5
const2_np = np.random.randn(S, S)
const2_onnx = onnx.helper.make_tensor("const2",
                                      onnx.onnx_pb2.TensorProto.FLOAT,
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
    ("test_constant_pad",
     N("Pad", mode='constant', value=1.2, paddings=[0, 0, 0, 0, 1, 2, 3, 4]),
     lambda x: np.pad(x,
                      pad_width=((0, 0), (0, 0), (1, 2), (3, 4)),
                      mode='constant',
                      constant_values=1.2),
     [(1, 3, L, M)]),
    ("test_refelction_pad",
     N("Pad", mode='reflect', paddings=[0, 0, 0, 0, 1, 1, 1, 1]),
     lambda x: np.pad(x,
                      pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
                      mode='reflect'),
     [(1, 3, L, M)]),
    ("test_edge_pad",
     N("Pad", mode='edge', paddings=[0, 0, 0, 0, 1, 1, 1, 1]),
     lambda x: np.pad(x,
                      pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
                      mode='edge'),
     [(1, 3, L, M)]),
    ("test_slice",
     N("Slice", axes=[0, 1], starts=[0, 0], ends=[3, M]),
     lambda x: x[0:3, 0:M], [(L, M, S)]),
    ("test_slice_neg",
     N("Slice", axes=[1], starts=[0], ends=[-1]),
     lambda x: x[:, 0:-1], [(L, M, S)]),
    ("test_slice_default_axes",
     N("Slice", starts=[0, 0, 3], ends=[L, M, 4]),
     lambda x: x[:, :, 3:4], [(L, M, S)]),
    # TODO: Add all the other operators
] + test_rnn.node_tests

model_tests = [
    ('test_bvlc_alexnet', 'bvlc_alexnet'),
    ('test_densenet121', 'densenet121'),
    ('test_inception_v1', 'inception_v1'),
    ('test_inception_v2', 'inception_v2'),
    ('test_resnet50', 'resnet50'),
    ('test_shufflenet', 'shufflenet'),
    ('test_squeezenet', 'squeezenet'),
    ('test_vgg16', 'vgg16'),
]

# Running vgg19 on Travis with Python 2 keeps getting OOM!
if not os.environ.get('TRAVIS'):
    model_tests.append(('test_vgg19', 'vgg19'))


class BackendTest(object):
    def __init__(self, backend, parent_module=None):
        class TestsContainer(unittest.TestCase):
            def setUp(self):
                np.random.seed(seed=0)

        self.backend = backend
        self._base_case = TestsContainer
        if parent_module:
            self._parent_module = parent_module
        # List of test cases to be applied on the parent scope
        # Example usage: globals().update(BackendTest(backend).test_cases)
        self.test_cases = {}

        for nt in node_tests:
            self._add_node_test(*nt)

        for gt in model_tests:
            self._add_model_test(*gt)

        # For backward compatibility - create a suite to aggregate them all
        self.tests = type(str('OnnxBackendTest'), (self._base_case,), {})
        for _, case in sorted(self.test_cases.items()):
            for name, func in sorted(case.__dict__.items()):
                if name.startswith('test_'):
                    setattr(self.tests, name, func)

    def _get_test_case(self, category):
        name = 'OnnxBackend{}Test'.format(category)
        if name not in self.test_cases:
            self.test_cases[name] = type(str(name), (self._base_case,), {})
            if self._parent_module:
                self.test_cases[name].__module__ = self._parent_module
        return self.test_cases[name]

    def _prepare_model(self, model_name):
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', '~/.onnx'))
        models_dir = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(
                model_name)
            with tempfile.NamedTemporaryFile(delete=True) as download_file:
                print('Start downloading model {} from {}'.format(model_name, url))
                urlretrieve(url, download_file.name)
                print('Done')
                with tarfile.open(download_file.name) as t:
                    t.extractall(models_dir)
        return model_dir

    def _add_test(self, test_case, name, test_func):
        # We don't prepend the 'test_' prefix to improve greppability
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        s = self._get_test_case(test_case)
        if hasattr(s, name):
            raise ValueError('Duplicated test name: {}'.format(name))
        setattr(s, name, test_func)

    def _add_model_test(self, test_name, model_name, device='CPU'):
        """
        Add A test for a single ONNX model against a reference implementation.
            test_name (string): Eventual name of the test.  Must be prefixed
                with 'test_'.
            model_name (string): The ONNX model's name
            inputs (list of ndarrays): inputs to the model
            outputs (list of ndarrays): outputs to the model
        """
        def run(test_self):
            if not self.backend.supports_device(device):
                raise unittest.SkipTest(
                    "Backend doesn't support device {}".format(device))
            model_dir = self._prepare_model(model_name)
            model_pb_path = os.path.join(model_dir, 'model.pb')
            model = onnx.load(model_pb_path)
            prepared_model = self.backend.prepare(model, device)

            for test_data_npz in glob.glob(os.path.join(model_dir, 'test_data_*.npz')):
                test_data = np.load(test_data_npz, encoding='bytes')
                inputs = list(test_data['inputs'])
                outputs = list(prepared_model.run(inputs))
                ref_outputs = test_data['outputs']
                test_self.assertEqual(len(ref_outputs), len(outputs))
                for i in range(len(outputs)):
                    np.testing.assert_almost_equal(
                        ref_outputs[i],
                        outputs[i],
                        decimal=4)

        self._add_test('Model', test_name, run)

    def _add_node_test(self, test_name, node_spec, ref, inputs, device='CPU'):
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
            if not self.backend.supports_device(device):
                raise unittest.SkipTest(
                    "Backend doesn't support device {}".format(device))
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
            outputs = self.backend.run_node(node_def, args, device)
            test_self.assertEqual(len(ref_outputs), len(outputs))
            for i in range(len(output_names)):
                np.testing.assert_almost_equal(
                    ref_outputs[i],
                    outputs[i],
                    decimal=4)

        self._add_test('Node', test_name, run)
