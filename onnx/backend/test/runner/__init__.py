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
from onnx import helper, numpy_helper
from six.moves.urllib.request import urlretrieve
from ..loader import load_node_tests, load_model_tests


class Runner(object):
    def __init__(self, backend, parent_module=None):
        class TestsContainer(unittest.TestCase):
            pass

        self.backend = backend
        self._base_case = TestsContainer
        self._parent_module = parent_module
        # List of test cases to be applied on the parent scope
        # Example usage: globals().update(BackendTest(backend).test_cases)
        self.test_cases = {}

        for nt in load_node_tests():
            self._add_node_test(nt)

        for gt in load_model_tests():
            self._add_model_test(gt)

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

    def _prepare_model_data(self, model_test):
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', '~/.onnx'))
        models_dir = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_test.model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(
                model_test.model_name)
            with tempfile.NamedTemporaryFile(delete=True) as download_file:
                print('Start downloading model {} from {}'.format(model_test.model_name, url))
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

        def add_test_for_device(device):
            @unittest.skipIf(
                not self.backend.supports_device(device),
                "Backend doesn't support device {}".format(device))
            def device_test_func(test_self):
                return test_func(test_self, device)
            setattr(s, '{}_{}'.format(name, device.lower()), device_test_func)

        for device in ['CPU', 'CUDA']:
            add_test_for_device(device)

    def _add_model_test(self, model_test):
        """
        Add A test for a single ONNX model against a reference implementation.
            test_name (string): Eventual name of the test.  Must be prefixed
                with 'test_'.
            model_name (string): The ONNX model's name
            inputs (list of ndarrays): inputs to the model
            outputs (list of ndarrays): outputs to the model
        """
        def run(test_self, device):
            model_dir = self._prepare_model_data(model_test)
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

        self._add_test('Model', model_test.name, run)

    def _add_node_test(self, node_test):
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
        def run(test_self, device):
            # TODO: In some cases we should generate multiple random inputs
            # and test (ala Hypothesis)
            np_inputs = [numpy_helper.to_array(tensor)
                         for tensor in node_test.inputs]
            ref_outputs = [numpy_helper.to_array(tensor)
                           for tensor in node_test.outputs]

            outputs = self.backend.run_node(node_test.node, np_inputs, device)
            test_self.assertEqual(len(ref_outputs), len(outputs))
            for i in range(len(outputs)):
                np.testing.assert_almost_equal(
                    ref_outputs[i],
                    outputs[i],
                    decimal=4)

        self._add_test('Node', node_test.name, run)
