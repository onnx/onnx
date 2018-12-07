from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest

import numpy as np  # type: ignore

import onnx
from onnx import helper, numpy_helper, NodeProto, ModelProto
from onnx.backend.base import Backend
from six.moves.urllib.request import urlretrieve
from ..loader import load_model_tests
from ..case.test_case import TestCase
from .item import TestItem
from typing import Optional, Pattern, Set, Dict, Text, Type, Sequence, Any, Callable, Union, Iterable, List


class BackendIsNotSupposedToImplementIt(unittest.SkipTest):
    pass


def retry_excute(times):  # type: (int) -> Callable[[Callable[..., Any]], Callable[..., Any]]
    assert times >= 1

    def wrapper(func):  # type: (Callable[..., Any]) -> Callable[..., Any]
        @functools.wraps(func)
        def wrapped(*args, **kwargs):  # type: (*Any, **Any) -> Any
            for i in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    print('{} times tried'.format(i))
                    if i == times:
                        raise
                    time.sleep(5 * i)
        return wrapped
    return wrapper


class Runner(object):

    def __init__(self, backend, parent_module=None):  # type: (Type[Backend], Optional[str]) -> None
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]

        # This is the source of the truth of all test functions.
        # Properties `test_cases`, `test_suite` and `tests` will be
        # derived from it.
        # {category: {name: func}}
        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        for rt in load_model_tests(kind='node'):
            self._add_model_test(rt, 'Node')

        for rt in load_model_tests(kind='real'):
            self._add_model_test(rt, 'Real')

        for rt in load_model_tests(kind='simple'):
            self._add_model_test(rt, 'Simple')

        for ct in load_model_tests(kind='pytorch-converted'):
            self._add_model_test(ct, 'PyTorchConverted')

        for ot in load_model_tests(kind='pytorch-operator'):
            self._add_model_test(ot, 'PyTorchOperator')

    def _get_test_case(self, name):  # type: (Text) -> Type[unittest.TestCase]
        test_case = type(str(name), (unittest.TestCase,), {})
        if self._parent_module:
            test_case.__module__ = self._parent_module
        return test_case

    def include(self, pattern):  # type: (Text) -> Runner
        self._include_patterns.add(re.compile(pattern))
        return self

    def exclude(self, pattern):  # type: (Text) -> Runner
        self._exclude_patterns.add(re.compile(pattern))
        return self

    def enable_report(self):  # type: () -> Runner
        import pytest  # type: ignore

        for category, items_map in self._test_items.items():
            for name, item in items_map.items():
                item.func = pytest.mark.onnx_coverage(item.proto, category)(item.func)
        return self

    @property
    def _filtered_test_items(self):  # type: () -> Dict[Text, Dict[Text, TestItem]]
        filtered = {}  # type: Dict[Text, Dict[Text, TestItem]]
        for category, items_map in self._test_items.items():
            filtered[category] = {}
            for name, item in items_map.items():
                if (self._include_patterns
                    and (not any(include.search(name)
                             for include in self._include_patterns))):
                    item.func = unittest.skip(
                        'no matched include pattern'
                    )(item.func)
                for exclude in self._exclude_patterns:
                    if exclude.search(name):
                        item.func = unittest.skip(
                            'matched exclude pattern "{}"'.format(
                                exclude.pattern)
                        )(item.func)
                filtered[category][name] = item
        return filtered

    @property
    def test_cases(self):  # type: () -> Dict[str, Type[unittest.TestCase]]
        '''
        List of test cases to be applied on the parent scope
        Example usage:
            globals().update(BackendTest(backend).test_cases)
        '''
        test_cases = {}
        for category, items_map in self._filtered_test_items.items():
            test_case_name = str('OnnxBackend{}Test').format(category)
            test_case = self._get_test_case(test_case_name)
            for name, item in sorted(items_map.items()):
                setattr(test_case, name, item.func)
            test_cases[test_case_name] = test_case
        return test_cases

    @property
    def test_suite(self):  # type: () -> unittest.TestSuite
        '''
        TestSuite that can be run by TestRunner
        Example usage:
            unittest.TextTestRunner().run(BackendTest(backend).test_suite)
        '''
        suite = unittest.TestSuite()
        for case in sorted(self.test_cases.values()):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        return suite

    # For backward compatibility (we used to expose `.tests`)
    @property
    def tests(self):  # type: () -> Type[unittest.TestCase]
        '''
        One single unittest.TestCase that hosts all the test functions
        Example usage:
            onnx_backend_tests = BackendTest(backend).tests
        '''
        tests = self._get_test_case('OnnxBackendTest')
        for items_map in sorted(self._filtered_test_items.values()):
            for name, item in sorted(items_map.items()):
                setattr(tests, name, item.func)
        return tests

    @staticmethod
    def _assert_similar_outputs(ref_outputs, outputs):  # type: (Sequence[Any], Sequence[Any]) -> None
        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype, outputs[i].dtype)
            np.testing.assert_allclose(
                ref_outputs[i],
                outputs[i],
                rtol=1e-3,
                atol=1e-7)

    @staticmethod
    @retry_excute(3)
    def _download_model(model_test, model_dir, models_dir):  # type: (TestCase, Text, Text) -> None
        # On Windows, NamedTemporaryFile can not be opened for a
        # second time
        download_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            download_file.close()
            print('Start downloading model {} from {}'.format(
                model_test.model_name,
                model_test.url))
            urlretrieve(model_test.url, download_file.name)
            print('Done')
            with tarfile.open(download_file.name) as t:
                t.extractall(models_dir)
        except Exception as e:
            print('Failed to prepare data for model {}: {}'.format(
                model_test.model_name, e))
            raise
        finally:
            os.remove(download_file.name)

    @staticmethod
    def _prepare_model_data(model_test):  # type: (TestCase) -> Text
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
        models_dir = os.getenv('ONNX_MODELS',
                               os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_test.model_name)  # type: Text
        if not os.path.exists(os.path.join(model_dir, 'model.onnx')):
            if os.path.exists(model_dir):
                bi = 0
                while True:
                    dest = '{}.old.{}'.format(model_dir, bi)
                    if os.path.exists(dest):
                        bi += 1
                        continue
                    shutil.move(model_dir, dest)
                    break
            os.makedirs(model_dir)

            Runner._download_model(model_test=model_test, model_dir=model_dir, models_dir=models_dir)
        return model_dir

    def _add_test(self,
                  category,  # type: Text
                  test_name,  # type: Text
                  test_func,  # type: Callable[..., Any]
                  report_item,  # type: List[Optional[Union[ModelProto, NodeProto]]]
                  devices=('CPU', 'CUDA'),  # type: Iterable[Text]
                  ):  # type: (...) -> None
        # We don't prepend the 'test_' prefix to improve greppability
        if not test_name.startswith('test_'):
            raise ValueError(
                'Test name must start with test_: {}'.format(test_name))

        def add_device_test(device):  # type: (Text) -> None
            device_test_name = '{}_{}'.format(test_name, device.lower())
            if device_test_name in self._test_items[category]:
                raise ValueError(
                    'Duplicated test name "{}" in category "{}"'.format(
                        device_test_name, category))

            @unittest.skipIf(  # type: ignore
                not self.backend.supports_device(device),
                "Backend doesn't support device {}".format(device))
            @functools.wraps(test_func)
            def device_test_func(*args, **kwargs):  # type: (*Any, **Any) -> Any
                try:
                    return test_func(*args, device=device, **kwargs)
                except BackendIsNotSupposedToImplementIt as e:
                    # hacky verbose reporting
                    if '-v' in sys.argv or '--verbose' in sys.argv:
                        print('Test {} is effectively skipped: {}'.format(
                            device_test_name, e))

            self._test_items[category][device_test_name] = TestItem(
                device_test_func, report_item)

        for device in devices:
            add_device_test(device)

    def _add_model_test(self, model_test, kind):  # type: (TestCase, Text) -> None
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run(test_self, device):  # type: (Any, Text) -> None
            if model_test.model_dir is None:
                model_dir = Runner._prepare_model_data(model_test)
            else:
                model_dir = model_test.model_dir
            model_pb_path = os.path.join(model_dir, 'model.onnx')
            model = onnx.load(model_pb_path)
            model_marker[0] = model
            if hasattr(self.backend, 'is_compatible') \
               and callable(self.backend.is_compatible) \
               and not self.backend.is_compatible(model):
                raise unittest.SkipTest('Not compatible with backend')
            prepared_model = self.backend.prepare(model, device)
            assert prepared_model is not None

            # TODO after converting all npz files to protobuf, we can delete this.
            for test_data_npz in glob.glob(
                    os.path.join(model_dir, 'test_data_*.npz')):
                test_data = np.load(test_data_npz, encoding='bytes')
                inputs = list(test_data['inputs'])
                outputs = list(prepared_model.run(inputs))
                ref_outputs = test_data['outputs']
                self._assert_similar_outputs(ref_outputs, outputs)

            for test_data_dir in glob.glob(
                    os.path.join(model_dir, "test_data_set*")):
                inputs = []
                inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
                for i in range(inputs_num):
                    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
                    tensor = onnx.TensorProto()
                    with open(input_file, 'rb') as f:
                        tensor.ParseFromString(f.read())
                    inputs.append(numpy_helper.to_array(tensor))
                ref_outputs = []
                ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
                for i in range(ref_outputs_num):
                    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
                    tensor = onnx.TensorProto()
                    with open(output_file, 'rb') as f:
                        tensor.ParseFromString(f.read())
                    ref_outputs.append(numpy_helper.to_array(tensor))
                outputs = list(prepared_model.run(inputs))
                self._assert_similar_outputs(ref_outputs, outputs)

        self._add_test(kind + 'Model', model_test.name, run, model_marker)
