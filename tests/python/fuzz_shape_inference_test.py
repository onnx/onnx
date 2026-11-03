# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import runpy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import onnx
from onnx import defs, helper

FUZZ_DIR = Path(__file__).resolve().parents[2] / "onnx" / "fuzz"


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


@pytest.mark.parametrize("version", range(8, 16))
def test_raw_model_last_byte(harness, monkeypatch, version: int) -> None:
    model = helper.make_model(
        helper.make_graph([], "raw", [], []),
        opset_imports=[helper.make_opsetid("", version)],
    )
    infer = Mock()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    harness["TestOneInput"](model.SerializeToString())
    infer.assert_called_once_with(
        model, strict_mode=bool(version & 1), check_type=bool(version & 2)
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
    fdp = SimpleNamespace(
        ConsumeIntInRange=Mock(return_value=index),
        ConsumeBool=Mock(side_effect=[True, custom_producer]),
    )
    harness["atheris"].FuzzedDataProvider.return_value = fdp
    infer = Mock()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    harness["TestOneInput"](b"\x00\x04")
    infer.assert_called_once()
    model = infer.call_args.args[0]
    target = model.graph.node[-1]
    assert (target.domain, target.op_type) == (domain, name)
    assert {op.domain: op.version for op in model.opset_import}[domain] == version
    if custom_producer:
        onnx.checker.check_model(model)
        assert list(model.graph.node[0].output) == list(target.input)
        assert not model.graph.value_info
    else:
        assert all(not value.HasField("type") for value in model.graph.input)
