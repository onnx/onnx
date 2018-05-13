from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

from ..case.test_case import TestCase
from typing import List, Text, Optional

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(os.path.dirname(__file__))),
    'data')


def load_model_tests(
    data_dir=DATA_DIR,  # type: Text
    kind=None,  # type: Optional[Text]
):  # type: (...) -> List[TestCase]
    '''Load model test cases from on-disk data files.
    '''

    supported_kinds = os.listdir(data_dir)
    if kind not in supported_kinds:
        raise ValueError("kind must be one of {}".format(supported_kinds))

    testcases = []

    kind_dir = os.path.join(data_dir, kind)
    for test_name in os.listdir(kind_dir):
        case_dir = os.path.join(kind_dir, test_name)
        # skip the non-dir files, such as generated __init__.py.
        if not os.path.isdir(case_dir):
            continue
        if os.path.exists(os.path.join(case_dir, 'model.onnx')):
            url = None
            model_name = test_name[len('test_')]
            model_dir = case_dir  # type: Optional[Text]
        else:
            with open(os.path.join(case_dir, 'data.json')) as f:
                data = json.load(f)
                url = data['url']
                model_name = data['model_name']
                model_dir = None
        testcases.append(
            TestCase(
                name=test_name,
                url=url,
                model_name=model_name,
                model_dir=model_dir,
                model=None,
                data_sets=None,
                kind=kind,
            ))

    return testcases
