from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import importlib
import os
import sys

from . import base
from .base import TestCase


def _find_submodules():
    return [fn[:-3] for fn in
            os.listdir(os.path.dirname(os.path.realpath(__file__)))
            if fn.endswith('.py') if fn != '__init__.py']

def _import_submodule(name):
    cur_module = sys.modules[__name__]
    return importlib.import_module('{}.{}'.format(
        cur_module.__name__, name))


def _import_all_submodules():
    for m in _find_submodules():
        _import_submodule(m)


def collect_testcases():
    _import_all_submodules()
    return base.TestCases


def collect_snippets():
    _import_all_submodules()
    return base.Snippets
