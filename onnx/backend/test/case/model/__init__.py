from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import onnx.defs
from ..utils import import_recursive
from ..test_case import TestCase

_SimpleModelTestCases = []


def expect(model, inputs, outputs, name=None):
    name = name or model.graph.name
    _SimpleModelTestCases.append(
        TestCase(
            name=name,
            model_name=model.graph.name,
            url=None,
            model_dir=None,
            model=model,
            data_sets=[(inputs, outputs)],
            kind='simple',
        ))


BASE_URL = 'https://s3.amazonaws.com/download.onnx/models/opset_{}'.format(
    onnx.defs.onnx_opset_version())


def collect_testcases():
    '''Collect model test cases defined in python/numpy code and in model zoo.
    '''

    real_model_testcases = []

    model_tests = [
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

    for test_name, model_name in model_tests:
        url = '{}/{}.tar.gz'.format(BASE_URL, model_name)
        real_model_testcases.append(TestCase(
            name=test_name,
            model_name=model_name,
            url=url,
            model_dir=None,
            model=None,
            data_sets=None,
            kind='real',
        ))

    import_recursive(sys.modules[__name__])

    return real_model_testcases + _SimpleModelTestCases
