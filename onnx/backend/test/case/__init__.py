# SPDX-License-Identifier: Apache-2.0

import sys

from .base import Snippets
from .utils import import_recursive
from typing import Dict, List, Tuple


def collect_snippets() -> Dict[str, List[Tuple[str, str]]]:
    import_recursive(sys.modules[__name__])
    return Snippets
