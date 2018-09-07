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

    def add_model(self, model, bucket, is_model):  # type: (onnx.ModelProto, Text, bool) -> None
        self.add_graph(model.graph, bucket)
        # Only add model if name does not start with test
        if is_model:
            self.models[bucket][model.graph.name].add(model)

    def add_proto(self, proto, bucket, is_model):  # type: (onnx.ModelProto, Text, bool) -> None
        assert isinstance(proto, onnx.ModelProto)
        self.add_model(proto, bucket, is_model)

    def report_text(self, writer):  # type: (IO[Text]) -> None
        writer.write('---------- onnx coverage: ----------\n')
        writer.write('Operators (passed/loaded/total): {}/{}/{}\n'.format(
            len(self.buckets['passed']),
            len(self.buckets['loaded']),
            len(_all_schemas)))
        writer.write('------------------------------------\n')

        rows = []
        passed = []
        all_ops = []  # type: List[Text]
        experimental = []  # type: List[Text]
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
            self.report_csv(all_ops, passed, experimental)

    # This function writes the coverage report to a set of CSV files for
    # the Backend Scoreboard (onnx.ai/backend-scoreboard). To enable this
    # feature, set a CSVDIR environment variable locally with the directory
    # where you would like the files to be written, relative to the
    # directory from which you're running pytest.  The format of the CSV
    # files is a column naming each op or model and columns for each
    # backend with indications of whether the tests passed or failed for
    # each row.
    def report_csv(self, all_ops, passed, experimental):  # type: (List[Text], List[Optional[Text]], List[Text]) -> None
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
        existing_nodes = OrderedDict()  # type: OrderedDict[Text, Dict[str, str]]
        existing_models = OrderedDict()  # type: OrderedDict[Text, Dict[str, str]]
        frameworks = []  # type: List[str]
        if os.path.isfile(nodes_path):
            with open(nodes_path, 'r') as nodes_file:
                reader = csv.DictReader(nodes_file)
                frameworks = list(reader.fieldnames)
                for row in reader:
                    op = row[str('Op')]
                    del row[str('Op')]
                    existing_nodes[str(op)] = row
        if os.path.isfile(models_path):
            with open(models_path, 'r') as models_file:
                reader = csv.DictReader(models_file)
                for row in reader:
                    model = row[str('Model')]
                    del row[str('Model')]
                    existing_models[str(model)] = row
        backend = os.environ.get(str('BACKEND'))
        other_frameworks = frameworks[1:]
        with open(nodes_path, 'w') as nodes_file:
            if str('Op') not in frameworks:
                frameworks.append(str('Op'))
            if backend not in frameworks:
                frameworks.append(str(backend))
            else:
                other_frameworks.remove(str(backend))
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
                        existing_nodes[node_name][other_framework] = str("Skipped!")
                if node in passed:
                    existing_nodes[node_name][str(backend)] = str("Passed!")
                else:
                    existing_nodes[node_name][str(backend)] = str("Failed!")
            summaries = dict()  # type: Dict[Any, Any]
            if "Summary" in existing_nodes:
                summaries = existing_nodes["Summary"]
                del existing_nodes["Summary"]
            summaries[str(backend)] = \
                "{}/{} node tests passed".format(len(passed), len(all_ops))
            summaries['Op'] = 'Summary'
            for node in existing_nodes:
                existing_nodes[node][str('Op')] = str(node)
                node_writer.writerow(existing_nodes[node])
            node_writer.writerow(summaries)
        with open(models_path, 'w') as models_file:
            frameworks[0] = str("Model")
            model_writer = csv.DictWriter(models_file, fieldnames=frameworks)
            model_writer.writeheader()
            # Consider both buckets
            num_models = 0
            for bucket in self.models:
                for model in self.models[bucket]:  # type: ignore
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
                            existing_models[model][other_framework] = str("Skipped!")
                    existing_models[model][str(backend)] = str("{}/{} nodes covered: {}"
                        .format(num_covered, len(self.models[bucket][model]
                            .node_coverages), msg))
            summaries.clear()
            if "Summary" in existing_models:
                summaries = existing_models["Summary"]
                del existing_models["Summary"]
            if str(backend) in summaries:
                del summaries[str(backend)]
            summaries[str(backend)] = "{}/{} model tests passed" \
                .format(len(self.models['passed']), num_models)
            summaries['Model'] = 'Summary'
            for model in existing_models:  # type: ignore
                existing_models[model][str('Model')] = model
                model_writer.writerow(existing_models[model])
            model_writer.writerow(summaries)
        with open(os.path.join(str(os.environ.get('CSVDIR')),  # type: ignore
                'metadata.csv'), 'w') as metadata_file:  # type: ignore
            metadata_writer = csv.writer(metadata_file)
            metadata_writer.writerow(["Latest Update", datetime.datetime.now().isoformat().replace('T', ' ')])
