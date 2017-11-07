from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import importlib
import pkgutil
import sys

from . import base
from .base import TestCase


def _import_recursive(package):
    """
    Takes a package and imports all modules underneath it
    """

    pkg_dir = package.__path__
    module_location = package.__name__
    for (_module_loader, name, ispkg) in pkgutil.iter_modules(pkg_dir):
        module_name = "{}.{}".format(module_location, name)  # Module/package
        module = importlib.import_module(module_name)
        if ispkg:
            import_recursive(module)


def collect_testcases():
    _import_recursive(sys.modules[__name__])
    return base.TestCases


def collect_snippets():
    _import_recursive(sys.modules[__name__])
    return base.Snippets
