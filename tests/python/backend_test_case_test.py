# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import pytest

import onnx.backend.test.runner as runner_module
from onnx import checker
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.case import node
from onnx.backend.test.loader import load_node_model_tests
from onnx.backend.test.runner import Runner
from onnx.reference import ReferenceEvaluator


@pytest.fixture
def make_cases(monkeypatch: pytest.MonkeyPatch):
    # Reuse actual node fixtures without changing the process-wide registry.
    sources = {case.name: case for case in load_node_model_tests()}
    monkeypatch.setattr(node, "_NodeTestCases", [])
    monkeypatch.setattr(node, "_existing_names", {})
    monkeypatch.setattr(node, "_TargetOpType", None)

    def make(source_name: str, **tolerances: float):
        source = sources[source_name]
        inputs, outputs = source.data_sets[0]
        node.expect(
            source.model.graph.node[0],
            inputs,
            outputs,
            name=f"{source_name}_custom_tolerance",
            opset_imports=list(source.model.opset_import),
            doc_string="Tolerance test model",
            **tolerances,
        )
        return node._NodeTestCases

    return make


@pytest.mark.parametrize("source_name", ["test_dynamicquantizelinear", "test_celu"])
@pytest.mark.parametrize(
    "tolerances, expected",
    [
        ({}, (1e-3, 1e-7)),
        ({"rtol": 0.0}, (0.0, 1e-7)),
        ({"atol": 0.0}, (1e-3, 0.0)),
        ({"rtol": 0.125, "atol": 0.25}, (0.125, 0.25)),
    ],
)
def test_expect_tolerances_propagate_to_function_expansions(
    make_cases,
    source_name: str,
    tolerances: dict[str, float],
    expected: tuple[float, float],
) -> None:
    cases = make_cases(source_name, **tolerances)
    # DynamicQuantizeLinear has a static body; Celu uses a context-dependent body.
    assert len(cases) >= 2
    assert "_expanded" not in cases[0].name
    assert all("_expanded" in case.name for case in cases[1:])
    for case in cases:
        assert (case.rtol, case.atol) == expected
        assert case.model.doc_string == "Tolerance test model"
        checker.check_model(case.model)
        inputs, outputs = case.data_sets[0]
        evaluator = ReferenceEvaluator(case.model)
        actual = evaluator.run(
            None, dict(zip(evaluator.input_names, inputs, strict=True))
        )
        Runner.assert_similar_outputs(outputs, actual, rtol=case.rtol, atol=case.atol)


@pytest.mark.parametrize(
    "source_name, tolerances, delta, override, passes",
    [
        ("test_dynamicquantizelinear", {"rtol": 0.0, "atol": 1.0}, 0, {}, True),
        ("test_dynamicquantizelinear", {"rtol": 0.0, "atol": 1.0}, 1, {}, True),
        ("test_dynamicquantizelinear", {"rtol": 0.0, "atol": 1.0}, -1, {}, True),
        ("test_dynamicquantizelinear", {"rtol": 0.0, "atol": 1.0}, 2, {}, False),
        ("test_dynamicquantizelinear", {"rtol": 0.0, "atol": 1.0}, -2, {}, False),
        (
            "test_dynamicquantizelinear",
            {"rtol": 0.0, "atol": 1.0},
            1,
            {"atol": 0.0},
            False,
        ),
        ("test_celu", {"rtol": 0.125, "atol": 0.0}, 0.0625, {}, True),
        ("test_celu", {"rtol": 0.125, "atol": 0.0}, 0.25, {}, False),
        ("test_celu", {"rtol": 0.125, "atol": 0.0}, 0.0625, {"rtol": 0.0}, False),
    ],
)
def test_runner_uses_case_tolerances_and_backend_overrides(
    make_cases,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    tolerances: dict[str, float],
    delta: float,
    override: dict[str, float],
    passes: bool,
) -> None:
    cases = make_cases(source_name, **tolerances)

    class PerturbedReferenceRep(BackendRep):
        def __init__(self, model):
            self.evaluator = ReferenceEvaluator(model)

        def run(self, inputs, **kwargs):  # noqa: ARG002
            values = self.evaluator.run(
                None, dict(zip(self.evaluator.input_names, inputs, strict=True))
            )
            values[0] = values[0].copy()
            # The quantized fixture starts at 153. Convert to a Python scalar
            # before adding signed deltas to avoid uint8 overflow in NumPy 2.
            values[0].flat[0] = values[0].flat[0].item() + delta
            return values

    class PerturbedReferenceBackend(Backend):
        @classmethod
        def prepare(cls, model, device="CPU", **kwargs):  # noqa: ARG003
            return PerturbedReferenceRep(model)

        @classmethod
        def supports_device(cls, device: str) -> bool:
            return device == "CPU"

    # Exercise the real runner while keeping unrelated models out of this test.
    monkeypatch.setattr(
        runner_module, "load_model_tests", lambda kind: cases if kind == "node" else []
    )
    runner = Runner(
        PerturbedReferenceBackend,
        test_kwargs={case.name: override for case in cases},
    )
    runner.include(r".*_cpu$")
    result = unittest.TestResult()
    runner.test_suite.run(result)
    assert result.testsRun - len(result.skipped) == len(cases)
    assert not result.errors, result.errors
    assert len(result.failures) == (0 if passes else len(cases)), result.failures
    for _, failure in result.failures:
        assert "Not equal to tolerance" in failure
