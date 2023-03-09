# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple

from onnx import FunctionProto, ValueInfoProto
from onnx.helper import (
    make_function,
    make_graph,
    make_node,
    make_opsetid,
)
from onnx.npx.npx_constants import FUNCTION_DOMAIN


def get_function_implementation(
    domop: Tuple[str, str],
    node_inputs: List[str],
    node_outputs: List[str],
    opsets: Dict[str, int],
    **kwargs: Any,
) -> FunctionProto:
    """
    Returns a :epkg:`FunctionProto` for a specific proto.

    :param domop: domain, function
    :param node_inputs: list of input names
    :param node_outputs: list of output names
    :param opsets: available opsets
    :kwargs: any other parameters
    :return: FunctionProto
    """
    if domop[0] != FUNCTION_DOMAIN:
        raise ValueError(
            f"This function only considers function for domain "
            f"{FUNCTION_DOMAIN!r} not {domop[0]!r}."
        )
    if domop[1] == "CDist":
        return _get_cdist_implementation(node_inputs, node_outputs, opsets, **kwargs)
    raise ValueError(f"Unable to return an implementation of function {domop!r}.")


def _get_cdist_implementation(
    node_inputs: List[str],
    node_outputs: List[str],
    opsets: Dict[str, int],
    **kwargs: Any,
) -> FunctionProto:
    """
    Returns the CDist implementation as a function.
    """
    if len(node_inputs) != 2:
        raise ValueError(f"cdist has two inputs not {len(node_inputs)}.")
    if len(node_outputs) != 1:
        raise ValueError(f"cdist has one outputs not {len(node_outputs)}.")
    if opsets is None:
        raise ValueError("opsets cannot be None.")
    if "" not in opsets:
        raise ValueError(
            "Opsets for domain '' must be specified but opsets={opsets!r}."
        )
    if set(kwargs) != {"metric"}:
        raise ValueError(f"kwargs={kwargs} must contain metric and only metric.")
    metric = kwargs["metric"]
    if opsets is not None and "com.microsoft" in opsets:
        node = make_node(
            "CDist", ["xa", "xb"], ["z"], domain="com.microsoft", metric=metric
        )
        return make_function(
            "npx",
            f"CDist_{metric}",
            ["xa", "xb"],
            ["z"],
            [node],
            [make_opsetid("com.microsoft", 1)],
        )

    if metric in ("euclidean", "sqeuclidean"):
        # subgraph
        nodes = [
            make_node("Sub", ["next", "next_in"], ["diff"]),
            make_node("Constant", [], ["axis"], value_ints=[1]),
            make_node("ReduceSumSquare", ["diff", "axis"], ["scan_out"], keepdims=0),
            make_node("Identity", ["next_in"], ["next_out"]),
        ]

        def make_value(name):
            value = ValueInfoProto()
            value.name = name
            return value

        graph = make_graph(
            nodes,
            "loop",
            [make_value("next_in"), make_value("next")],
            [make_value("next_out"), make_value("scan_out")],
        )

        scan = make_node(
            "Scan", ["xb", "xa"], ["next_out", "zout"], num_scan_inputs=1, body=graph
        )
        if metric == "euclidean":
            final = make_node("Sqrt", ["zout"], ["z"])
        else:
            final = make_node("Identity", ["zout"], ["z"])
        return make_function(
            "npx",
            f"CDist_{metric}",
            ["xa", "xb"],
            ["z"],
            [scan, final],
            [make_opsetid("", opsets[""])],
        )

    raise RuntimeError(
        f"There is no implementation for cdist and metric={metric!r} yet."
    )
