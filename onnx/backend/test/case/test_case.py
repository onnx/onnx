from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

TestCase = namedtuple('TestCase', [
    'name', 'model_name',
    'url',
    'model_dir',
    'model', 'data_sets',
    'kind',
])
