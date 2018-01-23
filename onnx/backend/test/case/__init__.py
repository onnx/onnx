from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

from .base import Snippets
from .utils import import_recursive


def collect_snippets():
    import_recursive(sys.modules[__name__])
    return Snippets
