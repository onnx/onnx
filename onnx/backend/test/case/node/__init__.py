from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import sys

from ..utils import import_recursive


TestCases = []

TestCase = namedtuple('TestCase', ['node', 'inputs', 'outputs', 'name'])


def expect(*args, **kwargs):
    TestCases.append(TestCase(*args, **kwargs))


def collect_testcases():
    '''Collect node test cases defined in python/numpy code.
    '''
    import_recursive(sys.modules[__name__])
    return TestCases
