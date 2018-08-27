from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import os
import csv

from tabulate import tabulate  # type: ignore

import onnx
from onnx import defs, helper, GraphProto
from typing import Optional, Text, Set, Dict, IO

_all_schemas = defs.get_all_schemas()

model_coverage_whitelist = set(
    ['bvlc_alexnet', 'densenet121', 'inception_v1', 'inception_v2',
     'resnet50', 'shufflenet', 'squeezenet_old', 'vgg19', 'zfnet'])


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


class ModelCoverage(object):
    def __init__(self):  # type: () -> None
        self.name = None  # type: Optional[Text]
        self.graph = None  # type: Optional[GraphProto]
        self.node_coverages = defaultdict(NodeCoverage)  # type: Dict[Text, NodeCoverage]

    def add(self, model):  # type: (onnx.ModelProto) -> None
        assert self.name in [None, model.graph.name]

        if self.name is None:
            self.name = model.graph.name
            assert self.name is not None
            self.graph = model.graph

        for node in model.graph.node:
            self.node_coverages[node.op_type].add(node)


class Coverage(object):
    def __init__(self):  # type: () -> None
        self.buckets = {
            'loaded': defaultdict(NodeCoverage),
            'passed': defaultdict(NodeCoverage),
        }  # type: Dict[Text, Dict[Text, NodeCoverage]]
        self.models = {
            'loaded': defaultdict(ModelCoverage),
            'passed': defaultdict(ModelCoverage),
        }  # type: Dict[Text, Dict[Text, ModelCoverage]]

    def add_node(self, node, bucket):  # type: (onnx.NodeProto, Text) -> None
        self.buckets[bucket][node.op_type].add(node)

    def add_graph(self, graph, bucket):  # type: (onnx.GraphProto, Text) -> None
        for node in graph.node:
            self.add_node(node, bucket)

    def add_model(self, model, bucket):  # type: (onnx.ModelProto, Text) -> None
        self.add_graph(model.graph, bucket)
        # Only add model if name does not start with test
        if model.graph.name in model_coverage_whitelist:
            self.models[bucket][model.graph.name].add(model)

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
        passed = []
        all_ops = []
        experimental = []
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
            passed.append(op_cov.op_type)
        writer.write(tabulate(
            rows,
            headers=['Operator', 'Attributes\n(name: #values)'],
            tablefmt='plain'))
        if os.environ.get(str('CSVDIR')) is not None:
            print("Writing csv file")
            for schema in _all_schemas:
                all_ops.append(schema.name)
                if schema.support_level == defs.OpSchema.SupportType.EXPERIMENTAL:
                    experimental.append(schema.name)
            all_ops.sort()
            with open(os.path.join(str(os.environ.get('CSVDIR')),  # type: ignore
                    os.environ.get('BACKEND') + '_nodes.csv'), 'w') as nodes_file:  # type: ignore
                node_writer = csv.writer(nodes_file)
                for node in all_ops:
                    node_name = node
                    if node in experimental:
                        node_name = node + ' (Experimental)'
                    if node in passed:
                        # u"\U0001F49A"
                        node_writer.writerow([node_name, "Passed!"])
                    else:
                        # u"\U0001F494"
                        node_writer.writerow([node_name, "Failed!"])
                node_writer.writerow(["Summary", "{}/{} node tests passed"
                    .format(len(passed), len(all_ops))])
            with open(os.path.join(str(os.environ.get('CSVDIR')),  # type: ignore
                    os.environ.get('BACKEND') + '_models.csv'), 'w') as models_file:  # type: ignore
                model_writer = csv.writer(models_file)
                # Consider both buckets
                for bucket in self.models:
                    for model in self.models[bucket]:
                        # Both analyze and run the model on the backend
                        num_covered = 0
                        for node in self.models[bucket][model].node_coverages:
                            if node in passed:
                                num_covered += 1
                        # TODO: Identify if there are models that are being
                        # skipped/not loaded, but that are in other frameworks
                        msg = "Passed!"
                        if bucket == 'loaded':
                            msg = "Failed!"
                        model_writer.writerow([model, "{}/{} nodes covered: {}"
                            .format(num_covered, len(self.models[bucket][model]
                                .node_coverages), msg)])
                model_writer.writerow(["Summary", "{}/{} model tests passed"
                    .format(len(self.models['passed']),
                        len(self.models['loaded']) + len(self.models['passed']))])
