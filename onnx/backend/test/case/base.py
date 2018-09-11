from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import inspect
from textwrap import dedent
from typing import Dict, Text, List, Tuple, Type, Sequence, Any

import numpy as np  # type: ignore
from six import add_metaclass


def process_snippet(op_name, name, export):  # type: (Text, Text, Any) -> Tuple[Text, Text]
    snippet_name = name[len('export_'):] or op_name.lower()
    source_code = dedent(inspect.getsource(export))
    # remove the function signature line
    lines = source_code.splitlines()
    assert lines[0] == '@staticmethod'
    assert lines[1].startswith('def export')
    return snippet_name, dedent("\n".join(lines[2:]))


Snippets = defaultdict(list)  # type: Dict[Text, List[Tuple[Text, Text]]]


class _Exporter(type):
    exports = defaultdict(list)  # type: Dict[Text, List[Tuple[Text, Text]]]

    def __init__(cls, name, bases, dct):  # type: (str, Tuple[Type[Any], ...], Dict[str, Any]) -> None
        for k, v in dct.items():
            if k.startswith('export'):
                if not isinstance(v, staticmethod):
                    raise ValueError(
                        'Only staticmethods could be named as export.*')
                export = getattr(cls, k)
                Snippets[name].append(process_snippet(name, k, export))
                # export functions should call expect and so populate
                # TestCases
                np.random.seed(seed=0)
                export()
        super(_Exporter, cls).__init__(name, bases, dct)


@add_metaclass(_Exporter)
class Base(object):
    pass
