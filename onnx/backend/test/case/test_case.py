# SPDX-License-Identifier: Apache-2.0


from collections import namedtuple

TestCase = namedtuple('TestCase', [
    'name',
    'model_name',
    'url',
    'model_dir',
    'model',
    'data_sets',
    'kind',
    'rtol',
    'atol',
])
# Tell PyTest this isn't a real test.
TestCase.__test__ = False
