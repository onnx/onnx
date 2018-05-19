from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import os

from tabulate import tabulate  # type: ignore

import onnx
from onnx import defs, helper
from typing import Optional, Text, Set, Dict, IO

_all_schemas = defs.get_all_schemas()


class AttrCoverage(object):
    def __init__(self):  # type: () -> None
        self.name = None  # type: Optional[Text]
        self.values = set()  # type: Set[Text]

    def add(self, attr):  # type: (onnx.AttributeProto) -> None
        assert self.name in [None, attr.name]
        self.name = attr.name
        value = helper.get_attribute_value(attr)
        # Turn list into tuple so we can put it into set
        # As value can be string, don't blindly turn `collections.Iterable`
        # into tuple.
        if isinstance(value, list):
            value = tuple(value)
        self.values.add(str(value))


class NodeCoverage(object):
    def __init__(self):  # type: () -> None
        self.op_type = None  # type: Optional[Text]
        self.attr_coverages = defaultdict(AttrCoverage)  # type: Dict[Text, AttrCoverage]

    def add(self, node):  # type: (onnx.NodeProto) -> None
        assert self.op_type in [None, node.op_type]

        if self.op_type is None:
            self.op_type = node.op_type
            assert self.op_type is not None
            self.schema = defs.get_schema(self.op_type)

        for attr in node.attribute:
            self.attr_coverages[attr.name].add(attr)


class Coverage(object):
    def __init__(self):  # type: () -> None
        self.buckets = {
            'loaded': defaultdict(NodeCoverage),
            'passed': defaultdict(NodeCoverage),
        }  # type: Dict[Text, Dict[Text, NodeCoverage]]

    def add_node(self, node, bucket):  # type: (onnx.NodeProto, Text) -> None
        self.buckets[bucket][node.op_type].add(node)

    def add_graph(self, graph, bucket):  # type: (onnx.GraphProto, Text) -> None
        for node in graph.node:
            self.add_node(node, bucket)

    def add_model(self, model, bucket):  # type: (onnx.ModelProto, Text) -> None
        self.add_graph(model.graph, bucket)

    def add_proto(self, proto, bucket):  # type: (onnx.ModelProto, Text) -> None
        assert isinstance(proto, onnx.ModelProto)
        self.add_model(proto, bucket)

    def report_text(self, writer):  # type: (IO[Text]) -> None
        writer.write('---------- onnx coverage: ----------\n')
        writer.write('Operators (passed/loaded/total): {}/{}/{}\n'.format(
            len(self.buckets['passed']),
            len(self.buckets['loaded']),
            len(_all_schemas)))
        writer.write('------------------------------------\n')

        rows = []
        for op_cov in self.buckets['passed'].values():
            covered_attrs = [
                '{}: {}'.format(attr_cov.name, len(attr_cov.values))
                for attr_cov in op_cov.attr_coverages.values()]
            uncovered_attrs = [
                '{}: 0'.format(attr)
                for attr in op_cov.schema.attributes
                if attr not in op_cov.attr_coverages
            ]
            attrs = sorted(covered_attrs) + sorted(uncovered_attrs)
            if attrs:
                attrs_column = os.linesep.join(attrs)
            else:
                attrs_column = 'No attributes'
            rows.append([op_cov.op_type, attrs_column])
        writer.write(tabulate(
            rows,
            headers=['Operator', 'Attributes\n(name: #values)'],
            tablefmt='plain'))
