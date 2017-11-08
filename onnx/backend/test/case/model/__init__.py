from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple


TestCase = namedtuple('TestCase', ['name', 'url', 'model_name'])

BASE_URL = 'https://s3.amazonaws.com/download.onnx/models'


def collect_testcases():
    res = []

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
        res.append(TestCase(test_name, url, model_name))

    return res
