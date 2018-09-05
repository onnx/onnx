from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import os
import csv
import datetime

from tabulate import tabulate  # type: ignore

import onnx
from onnx import defs, helper, GraphProto
from typing import Optional, Text, Set, Dict, IO, List, Any

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
            for schema in _all_schemas:
                if schema.domain == '' or schema.domain == 'ai.onnx':
                    all_ops.append(schema.name)
                    if schema.support_level == defs.OpSchema.SupportType.EXPERIMENTAL:
                        experimental.append(schema.name)
            all_ops.sort()
            nodes_path = os.path.join(str(os.environ.get('CSVDIR')),  # type: ignore
                    'nodes.csv')  # type: ignore
            models_path = os.path.join(str(os.environ.get('CSVDIR')),  # type: ignore
                    'models.csv')  # type: ignore
            existing_nodes = OrderedDict()  # type: OrderedDict[Text, Dict[Text, Text]]
            existing_models = OrderedDict()  # type: OrderedDict[Text, Dict[Text, Text]]
            frameworks = []  # type: List[Text]
            if os.path.isfile(nodes_path):
                with open(nodes_path, 'r') as nodes_file:
                    reader = csv.DictReader(nodes_file)
                    frameworks = list(reader.fieldnames)
                    for row in reader:
                        op = row['Op']
                        del row['Op']
                        existing_nodes[op] = row
            if os.path.isfile(models_path):
                with open(models_path, 'r') as models_file:
                    reader = csv.DictReader(models_file)
                    for row in reader:
                        model = row['Model']
                        del row['Model']
                        existing_models[model] = row
            backend = os.environ.get('BACKEND')
            other_frameworks = frameworks[1:]
            with open(nodes_path, 'w') as nodes_file:
                if 'Op' not in frameworks:
                    frameworks.append('Op')
                if backend not in frameworks:
                    frameworks.append(str(backend))
                else:
                    other_frameworks.remove(backend)
                node_writer = csv.DictWriter(nodes_file, fieldnames=frameworks)
                node_writer.writeheader()
                for node in all_ops:
                    node_name = node
                    if node in experimental:
                        node_name = node + ' (Experimental)'
                    if node_name not in existing_nodes:
                        # Also add Skipped for other nodes
                        existing_nodes[node_name] = OrderedDict()
                        for other_framework in other_frameworks:
                            existing_nodes[node_name][other_framework] = "Skipped!"
                    if node in passed:
                        existing_nodes[node_name][str(backend)] = "Passed!"
                    else:
                        existing_nodes[node_name][str(backend)] = "Failed!"
                summaries = dict()  # type: Dict[Any, Any]
                if "Summary" in existing_nodes:
                    summaries = existing_nodes["Summary"]
                    del existing_nodes["Summary"]
                summaries[str(backend)] = \
                    "{}/{} node tests passed".format(len(passed), len(all_ops))
                summaries['Op'] = 'Summary'
                for node in existing_nodes:
                    existing_nodes[node]['Op'] = node
                    node_writer.writerow(existing_nodes[node])
                node_writer.writerow(summaries)
            with open(models_path, 'w') as models_file:
                frameworks[0] = "Model"
                model_writer = csv.DictWriter(models_file, fieldnames=frameworks)
                model_writer.writeheader()
                # Consider both buckets
                num_models = 0
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
                            if model in self.models['passed']:
                                continue
                            msg = "Failed!"
                        num_models += 1
                        if model not in existing_models:
                            # Also add Skipped for other models
                            existing_models[model] = OrderedDict()
                            for other_framework in other_frameworks:
                                existing_models[model][other_framework] = "Skipped!"
                        existing_models[model][str(backend)] = "{}/{} nodes covered: {}" \
                            .format(num_covered, len(self.models[bucket][model]
                                .node_coverages), msg)
                summaries = dict()
                if "Summary" in existing_models:
                    summaries = existing_models["Summary"]
                    del existing_models["Summary"]
                summaries[str(backend)] = "{}/{} model tests passed" \
                    .format(len(self.models['passed']), num_models)
                summaries['Model'] = 'Summary'
                for model in existing_models:
                    existing_models[model]['Model'] = model
                    model_writer.writerow(existing_models[model])
                model_writer.writerow(summaries)
            with open(os.path.join(str(os.environ.get('CSVDIR')),  # type: ignore
                    'metadata.csv'), 'w') as metadata_file:  # type: ignore
                metadata_writer = csv.writer(metadata_file)
                metadata_writer.writerow(["Latest Update", datetime.datetime.now().isoformat().replace('T', ' ')])
