# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import runpy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import onnx
from onnx import defs, helper

FUZZ_DIR = Path(__file__).resolve().parents[2] / "onnx" / "fuzz"


class _ScriptedFDP:
    """Returns scripted values first, then the minimum of each range."""

    def __init__(self, ints=(), bools=()):
        self._ints = list(ints)
        self._bools = list(bools)

    def ConsumeIntInRange(self, low: int, high: int) -> int:
        value = self._ints.pop(0) if self._ints else low
        assert low <= value <= high
        return value

    def ConsumeBool(self) -> bool:
        return self._bools.pop(0) if self._bools else False


def _identity_infer() -> Mock:
    return Mock(side_effect=lambda model, **_: model)


@pytest.fixture
def harness(monkeypatch):
    # The harness dispatch can be checked on platforms without Atheris wheels.
    atheris = SimpleNamespace(
        instrument_imports=contextlib.nullcontext, FuzzedDataProvider=Mock()
    )
    monkeypatch.setitem(sys.modules, "atheris", atheris)
    return runpy.run_path(str(FUZZ_DIR / "fuzz_shape_inference.py"))


def test_seed_models_reach_inference(harness, monkeypatch, tmp_path) -> None:
    generator = runpy.run_path(str(FUZZ_DIR / "make_seed_corpus.py"))
    archives = [
        tmp_path / f"{name}.zip" for name in ("vc", "parser", "checker", "shape")
    ]
    monkeypatch.setattr(sys, "argv", ["make_seed_corpus.py", *map(str, archives)])
    assert generator["main"]() == 0

    infer = Mock(wraps=onnx.shape_inference.infer_shapes)
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    with zipfile.ZipFile(archives[-1]) as corpus:
        for name in corpus.namelist():
            harness["TestOneInput"](corpus.read(name))
    assert {call.args[0].graph.name for call in infer.call_args_list} == {
        "linear",
        "concat",
        "matmul",
        "reshape",
        "if_graph",
        "loop_graph",
    }
    harness["atheris"].FuzzedDataProvider.assert_not_called()


@pytest.mark.parametrize("version", range(8, 24))
def test_raw_model_last_byte(harness, monkeypatch, version: int) -> None:
    model = helper.make_model(
        helper.make_graph([], "raw", [], []),
        opset_imports=[helper.make_opsetid("", version)],
    )
    infer = _identity_infer()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    harness["TestOneInput"](model.SerializeToString())
    # The harness infers once more to check the result.
    assert infer.call_count == 2
    assert infer.call_args_list[0] == call(
        model,
        strict_mode=bool(version & 1),
        check_type=bool(version & 2),
        data_prop=bool(version & 8),
    )


@pytest.mark.parametrize("custom_producer", [False, True])
@pytest.mark.parametrize(
    ("domain", "name", "version"),
    [
        ("", "Relu", 13),
        ("", "RegexFullMatch", 20),
        ("ai.onnx.ml", "DictVectorizer", 1),
        ("ai.onnx.ml", "LabelEncoder", 1),
        ("ai.onnx.ml", "LabelEncoder", 2),
        ("ai.onnx.ml", "LabelEncoder", 4),
        ("ai.onnx.ml", "TreeEnsemble", 5),
    ],
)
def test_unknown_input_types_reach_inference(
    harness, monkeypatch, custom_producer: bool, domain: str, name: str, version: int
) -> None:
    index = next(
        (
            i
            for i, schema in enumerate(harness["_SCHEMAS"])
            if (schema.domain, schema.name, schema.since_version)
            == (domain, name, version)
        ),
        None,
    )
    if not defs.has(name, version, domain):
        pytest.skip("Schema is not registered in this build")
    assert index is not None, "Registered schema is missing from the generator"
    harness["atheris"].FuzzedDataProvider.return_value = _ScriptedFDP(
        ints=[index], bools=[True, custom_producer]
    )
    infer = _identity_infer()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    harness["TestOneInput"](b"\x00\x04")
    model = infer.call_args_list[0].args[0]
    target = model.graph.node[-1]
    assert (target.domain, target.op_type) == (domain, name)
    assert {op.domain: op.version for op in model.opset_import}[domain] == version
    if custom_producer:
        onnx.checker.check_model(model)
        assert list(model.graph.node[0].output) == list(target.input)
        assert not model.graph.value_info
    else:
        assert all(not value.HasField("type") for value in model.graph.input)


def _schema_index(harness, domain: str, name: str, version: int) -> int:
    return next(
        i
        for i, schema in enumerate(harness["_SCHEMAS"])
        if (schema.domain, schema.name, schema.since_version) == (domain, name, version)
    )


def test_input_kinds(harness, monkeypatch) -> None:
    # Reshape-14 has inputs (data, shape) and an optional allowzero attribute.
    index = _schema_index(harness, "", "Reshape", 14)
    typed, initializer = 1, 2
    harness["atheris"].FuzzedDataProvider.return_value = _ScriptedFDP(
        # schema, input count, data kind, data type kind (tensor), elem type,
        # rank, dim kinds/values, shape kind, initializer length, values.
        ints=[index, 2, typed, 0, 0, 2, 0, 3, 1, initializer, 2, 2, -1],
        # schema path, no custom producer, arbitrary data type, then INT64
        # initializer, allowzero.
        bools=[True, False, False, True, True],
    )
    infer = _identity_infer()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    harness["TestOneInput"](b"\x00\x04")
    model = infer.call_args_list[0].args[0]
    (data,) = model.graph.input
    assert data.name == "X0"
    dims = data.type.tensor_type.shape.dim
    assert dims[0].dim_value == 3
    assert dims[1].dim_param == "d1"
    (shape,) = model.graph.initializer
    assert shape.name == "X1"
    assert onnx.numpy_helper.to_array(shape).tolist() == [2, -1]
    assert [attr.name for attr in model.graph.node[0].attribute] == ["allowzero"]
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)


def test_omitted_optional_input(harness, monkeypatch) -> None:
    index = _schema_index(harness, "", "Clip", 13)
    harness["atheris"].FuzzedDataProvider.return_value = _ScriptedFDP(
        ints=[index, 2, 1, 0, 1, -1, 3], bools=[True, False]
    )
    infer = _identity_infer()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    harness["TestOneInput"](b"\x00\x04")
    node = infer.call_args_list[0].args[0].graph.node[0]
    assert list(node.input) == ["X0", ""]
