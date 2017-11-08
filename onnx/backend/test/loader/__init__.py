from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import functools
import glob
import json
import os
import tarfile
import tempfile
import unittest

import numpy as np
from six.moves.urllib.request import urlretrieve

import onnx
from onnx import helper, numpy_helper
from ..case.node import TestCase as NodeTestCase
from ..case.model import TestCase as ModelTestCase

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(os.path.dirname(__file__))),
    'data')


def load_node_tests(data_dir=os.path.join(DATA_DIR, 'node')):
    testcases = []

    for test_name in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, test_name)

        node = onnx.NodeProto()
        with open(os.path.join(case_dir, 'node.pb'), 'rb') as f:
            node.ParseFromString(f.read())

        inputs = []
        for input_file in sorted(
                glob.glob(os.path.join(case_dir, 'input_*.pb'))):
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inputs.append(tensor)

        outputs = []
        for output_file in sorted(
                glob.glob(os.path.join(case_dir, 'output_*.pb'))):
            tensor = onnx.TensorProto()
            with open(output_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            outputs.append(tensor)

        testcases.append(
            NodeTestCase(node, inputs, outputs, test_name))

    return testcases


def load_model_tests(data_dir=os.path.join(DATA_DIR, 'model')):
    testcases = []

    for test_name in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, test_name)
        with open(os.path.join(case_dir, 'data.json')) as f:
            data = json.load(f)
            url = data['url']
            model_name = data['model_name']
        testcases.append(
            ModelTestCase(test_name, url, model_name))

    return testcases


class BackendTest(object):

    def __init__(self, backend, parent_module=None)
        self.backend = backend
        self._parent_module = parent_module

        # This is the source of the truth of all test functions.
        # Properties `test_cases`, `test_suite` and `tests` will be
        # derived from it.
        # {category: {name: func}}
        self._test_funcs_map = defaultdict(dict)

        for nt in load_node_tests():
            self._add_node_test(nt)

        for gt in load_model_tests():
            self._add_model_test(gt)

    def _get_test_case(self, name):
        test_case = type(str(name), (unittest.TestCase,), {})
        if self._parent_module:
            test_case.__module__ = self._parent_module
        return test_case

    @property
    def test_cases(self):
        '''
        List of test cases to be applied on the parent scope
        Example usage:
            globals().update(BackendTest(backend).test_cases)
        '''
        test_cases = {}
        for category, funcs_map in self._test_funcs_map.items():
            test_case_name = 'OnnxBackend{}Test'.format(category)
            test_case = self._get_test_case(test_case_name)
            for name, func in sorted(funcs_map.items()):
                setattr(test_case, name, func)
            test_cases[test_case_name] = test_case
        return test_cases

    @property
    def test_suite(self):
        '''
        TestSuite that can be run by TestRunner
        Example usage:
            unittest.TextTestRunner().run(BackendTest(backend).test_suite)
        '''
        suite = unittest.TestSuite()
        for case in sorted(self.test_cases.values()):
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromTestCase(case))
        return suite

    # For backward compatibility - we used to expose `.tests`
    @property
    def tests(self):
        '''
        One single unittest.TestCase that hosts all the test functions
        Example usage:
            onnx_backend_tests = BackendTest(backend).tests
        '''
        tests = self._get_test_case('OnnxBackendTest')
        for _, funcs_map in sorted(self._test_funcs_map.values()):
            for name, func in sorted(funcs_map.items()):
                setattr(tests, name, func)
        return tests

    @staticmethod
    def _assert_similar_outputs(ref_outputs, outputs):
        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            np.testing.assert_allclose(
                ref_outputs[i],
                outputs[i],
                rtol=1e-3)

    def _prepare_model_data(self, model_test):
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', '~/.onnx'))
        models_dir = os.getenv('ONNX_MODELS',
                               os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_test.model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(
                model_test.model_name)
            with tempfile.NamedTemporaryFile(delete=True) as download_file:
                print('Start downloading model {} from {}'.format(
                    model_test.model_name, url))
                urlretrieve(url, download_file.name)
                print('Done')
                with tarfile.open(download_file.name) as t:
                    t.extractall(models_dir)
        return model_dir

    def _add_test(self, category, test_name, test_func):
        # We don't prepend the 'test_' prefix to improve greppability
        if not test_name.startswith('test_'):
            raise ValueError(
                'Test name must start with test_: {}'.format(test_name))

        def add_test_for_device(device):
            device_test_name = '{}_{}'.format(test_name, device.lower())
            if device_test_name in self._test_funcs_map[category]:
                raise ValueError(
                    'Duplicated test name "{}" in category "{}"'.format(
                        device_test_name, category))

            @unittest.skipIf(
                not self.backend.supports_device(device),
                "Backend doesn't support device {}".format(device))
            @functools.wraps(test_func)
            def device_test_func(test_self):
                return test_func(test_self, device)

            self._test_funcs_map[category][device_test_name] = device_test_func

        for device in ['CPU', 'CUDA']:
            add_test_for_device(device)

    def _add_model_test(self, model_test):
        def run(test_self, device):
            model_dir = self._prepare_model_data(model_test)
            model_pb_path = os.path.join(model_dir, 'model.pb')
            model = onnx.load(model_pb_path)
            prepared_model = self.backend.prepare(model, device)

            for test_data_npz in glob.glob(
                    os.path.join(model_dir, 'test_data_*.npz')):
                test_data = np.load(test_data_npz, encoding='bytes')
                inputs = list(test_data['inputs'])
                outputs = list(prepared_model.run(inputs))
                ref_outputs = test_data['outputs']
                self._assert_similar_outputs(ref_outputs, outputs)

        self._add_test('Model', model_test.name, run)

    def _add_node_test(self, node_test):
        def run(test_self, device):
            np_inputs = [numpy_helper.to_array(tensor)
                         for tensor in node_test.inputs]
            ref_outputs = [numpy_helper.to_array(tensor)
                           for tensor in node_test.outputs]

            outputs = self.backend.run_node(
                node_test.node, np_inputs, device)
            self._assert_similar_outputs(ref_outputs, outputs)

        self._add_test('Node', node_test.name, run)
