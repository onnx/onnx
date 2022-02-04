# SPDX-License-Identifier: Apache-2.0

import sys

from .base import Snippets
from .utils import import_recursive
from typing import Dict, Text, List, Tuple


def collect_snippets() -> Dict[Text, List[Tuple[Text, Text]]]:
    import_recursive(sys.modules[__name__])
    return Snippets
