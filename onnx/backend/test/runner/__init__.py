# SPDX-License-Identifier: Apache-2.0

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
from onnx import helper, numpy_helper, NodeProto, ModelProto, TypeProto
from onnx.backend.base import Backend
from urllib.request import urlretrieve
from ..loader import load_model_tests
from ..case.test_case import TestCase
from .item import TestItem
from typing import Optional, Pattern, Set, Dict, Text, Type, Sequence, Any, Callable, Union, Iterable, List


class BackendIsNotSupposedToImplementIt(unittest.SkipTest):
    pass


def retry_excute(times: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    assert times >= 1

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
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

    def __init__(self, backend: Type[Backend], parent_module: Optional[str] = None) -> None:
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns: Set[Pattern[Text]] = set()
        self._exclude_patterns: Set[Pattern[Text]] = set()
        self._xfail_patterns: Set[Pattern[Text]] = set()

        # This is the source of the truth of all test functions.
        # Properties `test_cases`, `test_suite` and `tests` will be
        # derived from it.
        # {category: {name: func}}
        self._test_items: Dict[Text, Dict[Text, TestItem]] = defaultdict(dict)

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

    def _get_test_case(self, name: Text) -> Type[unittest.TestCase]:
        test_case = type(str(name), (unittest.TestCase,), {})
        if self._parent_module:
            test_case.__module__ = self._parent_module
        return test_case

    # TODO: Use proper type annotation rather than string
    # once we drop Python 3.6 support. See
    # https://www.python.org/dev/peps/pep-0563/.
    def include(self, pattern: Text) -> 'Runner':
        self._include_patterns.add(re.compile(pattern))
        return self

    def exclude(self, pattern: Text) -> 'Runner':
        self._exclude_patterns.add(re.compile(pattern))
        return self

    def xfail(self, pattern: Text) -> 'Runner':
        self._xfail_patterns.add(re.compile(pattern))
        return self

    def enable_report(self) -> 'Runner':
        import pytest  # type: ignore

        for category, items_map in self._test_items.items():
            for name, item in items_map.items():
                item.func = pytest.mark.onnx_coverage(item.proto, category)(item.func)
        return self

    @property
    def _filtered_test_items(self) -> Dict[Text, Dict[Text, TestItem]]:
        filtered: Dict[Text, Dict[Text, TestItem]] = {}
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
                for xfail in self._xfail_patterns:
                    if xfail.search(name):
                        item.func = unittest.expectedFailure(item.func)
                filtered[category][name] = item
        return filtered

    @property
    def test_cases(self) -> Dict[str, Type[unittest.TestCase]]:
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
    def test_suite(self) -> unittest.TestSuite:
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
    def tests(self) -> Type[unittest.TestCase]:
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

    @classmethod
    def assert_similar_outputs(cls, ref_outputs: Sequence[Any], outputs: Sequence[Any], rtol: float, atol: float) -> None:
        np.testing.assert_equal(len(outputs), len(ref_outputs))
        for i in range(len(outputs)):
            if isinstance(outputs[i], (list, tuple)):
                for j in range(len(outputs[i])):
                    cls.assert_similar_outputs(ref_outputs[i][j], outputs[i][j], rtol, atol)
            else:
                np.testing.assert_equal(outputs[i].dtype, ref_outputs[i].dtype)
                if ref_outputs[i].dtype == np.object:
                    np.testing.assert_array_equal(outputs[i], ref_outputs[i])
                else:
                    np.testing.assert_allclose(
                        outputs[i],
                        ref_outputs[i],
                        rtol=rtol,
                        atol=atol)

    @classmethod
    @retry_excute(3)
    def download_model(cls, model_test: TestCase, model_dir: Text, models_dir: Text) -> None:
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

    @classmethod
    def prepare_model_data(cls, model_test: TestCase) -> Text:
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
        models_dir = os.getenv('ONNX_MODELS',
                               os.path.join(onnx_home, 'models'))
        model_dir: Text = os.path.join(models_dir, model_test.model_name)
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

            cls.download_model(model_test=model_test, model_dir=model_dir, models_dir=models_dir)
        return model_dir

    def _add_test(self,
                  category: Text,
                  test_name: Text,
                  test_func: Callable[..., Any],
                  report_item: List[Optional[Union[ModelProto, NodeProto]]],
                  devices: Iterable[Text] = ('CPU', 'CUDA'),
                  ) -> None:
        # We don't prepend the 'test_' prefix to improve greppability
        if not test_name.startswith('test_'):
            raise ValueError(
                'Test name must start with test_: {}'.format(test_name))

        def add_device_test(device: Text) -> None:
            device_test_name = '{}_{}'.format(test_name, device.lower())
            if device_test_name in self._test_items[category]:
                raise ValueError(
                    'Duplicated test name "{}" in category "{}"'.format(
                        device_test_name, category))

            @unittest.skipIf(  # type: ignore
                not self.backend.supports_device(device),
                "Backend doesn't support device {}".format(device))
            @functools.wraps(test_func)
            def device_test_func(*args: Any, **kwargs: Any) -> Any:
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

    def _add_model_test(self, model_test: TestCase, kind: Text) -> None:
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker: List[Optional[Union[ModelProto, NodeProto]]] = [None]

        def run(test_self: Any, device: Text) -> None:
            if model_test.model_dir is None:
                model_dir = self.prepare_model_data(model_test)
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
                self.assert_similar_outputs(ref_outputs, outputs,
                                            rtol=model_test.rtol,
                                            atol=model_test.atol)
            for test_data_dir in glob.glob(
                    os.path.join(model_dir, "test_data_set*")):
                inputs = []
                inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
                for i in range(inputs_num):
                    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
                    self._load_proto(input_file, inputs, model.graph.input[i].type)
                ref_outputs = []
                ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
                for i in range(ref_outputs_num):
                    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
                    self._load_proto(output_file, ref_outputs, model.graph.output[i].type)
                outputs = list(prepared_model.run(inputs))
                self.assert_similar_outputs(ref_outputs, outputs,
                                            rtol=model_test.rtol,
                                            atol=model_test.atol)

        self._add_test(kind + 'Model', model_test.name, run, model_marker)

    def _load_proto(self, proto_filename: Text, target_list: List[Union[np.ndarray, List[Any]]], model_type_proto: TypeProto) -> None:
        with open(proto_filename, 'rb') as f:
            protobuf_content = f.read()
            if model_type_proto.HasField('sequence_type'):
                sequence = onnx.SequenceProto()
                sequence.ParseFromString(protobuf_content)
                target_list.append(numpy_helper.to_list(sequence))
            elif model_type_proto.HasField('tensor_type'):
                tensor = onnx.TensorProto()
                tensor.ParseFromString(protobuf_content)
                target_list.append(numpy_helper.to_array(tensor))
            elif model_type_proto.HasField('optional_type'):
                optional = onnx.OptionalProto()
                optional.ParseFromString(protobuf_content)
                target_list.append(numpy_helper.to_optional(optional))
            else:
                print('Loading proto of that specific type (Map/Sparse Tensor) is currently not supported')
