# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto
from onnx.helper import (
    make_attribute,
    make_function,
    make_graph,
    make_node,
    make_operatorsetid,
    make_value_info,
)
from onnx.numpy_helper import from_array
from onnx.version_converter import convert_version


def rename_in_onnx_graph(
    graph: GraphProto, replacements: Dict[str, str]
) -> Optional[GraphProto]:
    """
    Renames input results in a GraphProto.

    :param graph: :epkg:`GraphProto`
    :param replacements: replacements `{ old_name: new_name }`
    :return: modified :epkg:`GraphProto` or None if no modifications
        were detected
    """

    def _process_attributes(attributes):
        atts = []
        modified = False
        for att in attributes:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                new_g = rename_in_onnx_graph(att.g, replacements)
                if new_g is None:
                    atts.append(att)
                    continue
                modified = True
                att = make_attribute(att.name, new_g)
            atts.append(att)
        return atts if modified else None

    set_rep = set(replacements)
    nodes = []
    modified = False
    for node in graph.node:
        if len(set(node.input) & set_rep) == 0:
            modified = True
            new_inputs = [replacements.get(i, i) for i in node.input]
            atts = _process_attributes(node.attribute) or node.attribute
            new_node = make_node(
                node.op_type, new_inputs, node.output, domain=node.domain
            )
            new_node.attribute.extend(atts)
            nodes.append(new_node)
            continue

        new_atts = _process_attributes(node.attribute)
        if new_atts is None:
            modified = True
            nodes.append(node)

    if not modified:
        return None

    if len(set(i.name for i in graph.input) & set_rep) == 0:
        return make_graph(nodes, graph.name, graph.input, graph.output)

    new_inputs = []
    for inp in graph.input:
        if inp.name in replacements:
            new = make_value_info(replacements.get(inp.name, inp.name), inp.t)  # type: ignore[attr-defined]
            new_inputs.append(new)  # type: ignore[arg-type]
            continue
        new_inputs.append(inp)  # type: ignore[arg-type]
    new_graph = make_graph(nodes, graph.name, new_inputs, graph.output)  # type: ignore[arg-type]
    return new_graph


def onnx_convert_model_for_opsets(
    model: ModelProto, target_opsets: Dict[str, int]
) -> ModelProto:
    """
    Checks the consistency of the model with the desired target_opsets.

    :param model: onnx model
    :param target_opsets: desired opsets `{ domain: version }`
    :return: modified model
    """
    if target_opsets is None:
        return model
    existing_opsets = {d.domain: d.version for d in model.opset_import}
    domains = []
    for domain, version in target_opsets.items():
        if domain not in existing_opsets:
            existing_opsets[domain] = version
            continue
        if existing_opsets[domain] == target_opsets[domain]:
            continue
        domains.append(
            (domain, existing_opsets.get(domain, None), target_opsets.get(domain, None))
        )
    if len(domains) == 1 and domains[0][0] == "":
        # Use the conversion.
        new_model = convert_version(model, domains[0][2])  # type: ignore[arg-type]
    elif len(domains) > 1:
        msg = ", ".join(
            f"domain={b!r}, from {before} -> {after}" for b, before, after in domains
        )
        raise RuntimeError(
            f"Unable to convert a model for the following domains {msg}."
        )
    else:
        new_model = model
    return new_model


def iter_nodes(nodes: Sequence[NodeProto]) -> Iterator[NodeProto]:
    """
    Iterates on all nodes within a graph and its subgraphs.
    """
    for node in nodes:
        yield node
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                for n in iter_nodes(att.g.node):
                    yield n


def onnx_model_to_function(
    onx: ModelProto,
    name: Optional[str] = None,
    domain: str = "custom",
    opset_imports: Optional[Dict[str, int]] = None,
    doc_string: Optional[str] = None,
) -> Tuple[FunctionProto, List[FunctionProto]]:
    """
    Converts an ONNX model into a function. The returned function
    has no attribute.
    :param onx: onnx model
    :param name: function name
    :param domain: function domain
    :param opset_imports: opset to import as a dictionary
        `{domain: version}`
    :param doc_string: doc string
    :param inputs2par: dictionary to move some inputs as attributes
        `{ name: None or default value }`
    :return: function, other functions
    .. warning::
        :epkg:`FunctionProto` does not support default values yet.
        They are ignored.
    """
    if isinstance(onx, ModelProto):
        if opset_imports is None:
            domains = {}
            for op in onx.opset_import:
                domains[op.domain] = op.version
            opset_imports = domains
        if doc_string is None:
            doc_string = onx.doc_string
        fp, lf = onnx_model_to_function(
            onx.graph,  # type: ignore[arg-type]
            name=name,
            domain=domain,
            opset_imports=opset_imports,
            doc_string=doc_string,
        )
        return fp, lf + list(onx.functions)

    if not isinstance(onx, GraphProto):
        raise TypeError(f"Unexpected type {type(onx)!r} for onx.")  # pragma: no cover

    if name is None:
        name = onx.name

    inputs = []
    outputs = [o.name for o in onx.output]
    attributes = []
    nodes = []
    for i in onx.input:
        inputs.append(i.name)

    if len(onx.initializer) > 0 or len(onx.sparse_initializer) > 0:
        # Needs to convert every initializer into Constant.
        csts = []
        for init in onx.initializer:
            value = from_array(init)
            n = make_node("Constant", [], [init.name], value=value)
            csts.append(n)
        for init in onx.sparse_initializer:
            value = from_array(init)
            n = make_node("Constant", [], [init.name], sparse_value=value)
            csts.append(n)
        nodes.extend(csts)

    nodes.extend(onx.node)

    # fixes domains
    opsets = {}
    for node in iter_nodes(nodes):
        if node.domain not in opsets:
            opsets[node.domain] = opset_imports.get(node.domain, 1)
    ops = [make_operatorsetid(k, v) for k, v in opsets.items()]

    return (
        make_function(
            domain,
            name,
            inputs,
            outputs,
            nodes,
            opset_imports=ops,
            doc_string=doc_string or "",
            attributes=attributes,
        ),
        [],
    )
