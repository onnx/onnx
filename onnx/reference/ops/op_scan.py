# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun, RefAttrName


class Scan(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        if not isinstance(self.body, RefAttrName):
            if not hasattr(self.body, "run"):
                raise RuntimeError(
                    f"Parameter 'body' must have a method 'run', type {type(self.body)}."
                )
            self.input_names = self.body.input_names
            self.output_names = self.body.output_names
        if not isinstance(self.num_scan_inputs, RefAttrName):
            if self.num_scan_inputs <= 0:
                raise RuntimeError(
                    f"Scan requires num_scan_inputs > 0, got {self.num_scan_inputs}."
                )
            if not isinstance(self.scan_input_directions, RefAttrName):
                in_dirs = self._resolve_int_list(
                    self.scan_input_directions,
                    self.num_scan_inputs,
                    "scan_input_directions",
                )
                self._validate_directions(in_dirs, "scan_input_directions")
            if not isinstance(self.scan_input_axes, RefAttrName):
                self._resolve_int_list(
                    self.scan_input_axes, self.num_scan_inputs, "scan_input_axes"
                )

    @staticmethod
    def _resolve_int_list(values, expected_len: int, attr_name: str) -> list[int]:
        if values is None or len(values) == 0:
            return [0] * expected_len
        if len(values) != expected_len:
            raise ValueError(
                f"Attribute {attr_name!r} has length {len(values)}, expected {expected_len}."
            )
        return [int(v) for v in values]

    @staticmethod
    def _validate_directions(directions: list[int], attr_name: str) -> None:
        for d in directions:
            if d not in (0, 1):
                raise ValueError(f"Invalid {attr_name} value {d}; expected 0 or 1.")

    @staticmethod
    def _normalize_axis(axis: int, rank: int, attr_name: str, index: int) -> int:
        if axis < -rank or axis >= rank:
            raise ValueError(
                f"{attr_name}[{index}]={axis} is out of bounds for tensor of rank {rank}."
            )
        return axis + rank if axis < 0 else axis

    def need_context(self) -> bool:
        """Scan body subgraphs may reference names from the enclosing graph
        (lexical capture, per the ONNX spec). Request the outer-scope
        context so those names resolve at body-evaluation time.
        """
        return True

    def _run(  # type: ignore[override]
        self,
        *args,
        context=None,
        body=None,
        num_scan_inputs=None,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        scan_output_directions=None,
        attributes=None,
        bindings=None,
    ):
        effective_body = self.body if body is None else body
        if not hasattr(effective_body, "run"):
            raise RuntimeError(
                f"Parameter 'body' must have a method 'run', type {type(effective_body)}."
            )
        effective_num_scan_inputs = (
            self.num_scan_inputs if num_scan_inputs is None else num_scan_inputs
        )
        if effective_num_scan_inputs <= 0:
            raise RuntimeError(
                f"Scan requires num_scan_inputs > 0, got {effective_num_scan_inputs}."
            )
        if effective_num_scan_inputs > len(args):
            raise ValueError(
                f"num_scan_inputs ({effective_num_scan_inputs}) cannot exceed "
                f"the number of inputs ({len(args)})."
            )

        num_loop_state_vars = len(args) - effective_num_scan_inputs
        input_names = effective_body.input_names
        output_names = effective_body.output_names
        if len(output_names) < num_loop_state_vars:
            raise ValueError(
                f"Scan body has {len(output_names)} outputs, expected at least "
                f"{num_loop_state_vars} loop-state outputs."
            )
        num_scan_outputs = len(output_names) - num_loop_state_vars

        effective_in_dirs = (
            self.scan_input_directions
            if scan_input_directions is None
            else scan_input_directions
        )
        effective_in_axes = (
            self.scan_input_axes if scan_input_axes is None else scan_input_axes
        )
        effective_out_dirs = (
            self.scan_output_directions
            if scan_output_directions is None
            else scan_output_directions
        )
        effective_out_axes = (
            self.scan_output_axes if scan_output_axes is None else scan_output_axes
        )

        input_directions = self._resolve_int_list(
            effective_in_dirs, effective_num_scan_inputs, "scan_input_directions"
        )
        self._validate_directions(input_directions, "scan_input_directions")
        input_axes = self._resolve_int_list(
            effective_in_axes, effective_num_scan_inputs, "scan_input_axes"
        )
        output_directions = self._resolve_int_list(
            effective_out_dirs, num_scan_outputs, "scan_output_directions"
        )
        self._validate_directions(output_directions, "scan_output_directions")
        output_axes = self._resolve_int_list(
            effective_out_axes, num_scan_outputs, "scan_output_axes"
        )

        state_names_in = input_names[:num_loop_state_vars]
        state_names_out = output_names[:num_loop_state_vars]
        scan_names_in = input_names[num_loop_state_vars:]
        scan_names_out = output_names[num_loop_state_vars:]
        scan_values = args[num_loop_state_vars:]
        states = list(args[:num_loop_state_vars])

        prepared_scan_inputs = []
        for i, (val, in_axis, in_dir) in enumerate(
            zip(scan_values, input_axes, input_directions, strict=True)
        ):
            arr = np.asarray(val)
            norm_in_axis = self._normalize_axis(in_axis, arr.ndim, "scan_input_axes", i)
            view = np.moveaxis(arr, norm_in_axis, 0)
            if in_dir == 1:
                view = view[::-1]
            prepared_scan_inputs.append(view)

        seq_lengths = [v.shape[0] for v in prepared_scan_inputs]
        max_iter = seq_lengths[0]
        if any(length != max_iter for length in seq_lengths):
            raise ValueError(
                f"Scan inputs must have matching sequence lengths along their scan axes, "
                f"got {seq_lengths}."
            )

        results: list[list[np.ndarray]] = [[] for _ in scan_names_out]

        for it in range(max_iter):
            # Seed with outer-scope values first; per-iteration state and
            # scan-slice inputs (added below) shadow any same-named outer
            # values, matching ONNX's lexical-capture semantics.
            inputs: dict = {}
            if context is not None:
                inputs.update(context)
            inputs.update(dict(zip(state_names_in, states, strict=False)))
            inputs.update(
                {
                    name: value[it]
                    for name, value in zip(
                        scan_names_in, prepared_scan_inputs, strict=False
                    )
                }
            )

            try:
                if effective_body is self.body:
                    outputs_list = self._run_body(
                        inputs, attributes=attributes, bindings=bindings
                    )
                else:
                    outputs_list = OpRun._evaluate_subgraph(
                        inputs, effective_body, attributes, bindings
                    )
            except TypeError as e:
                raise TypeError(
                    f"Unable to call 'run' for type '{type(effective_body)}'."
                ) from e

            outputs = dict(zip(output_names, outputs_list, strict=False))
            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                results[i].append(outputs[name])

        if max_iter == 0 and results:
            # Zero-trip Scan with scan outputs is not supported: the per-iteration
            # element shape/dtype cannot be reliably inferred without executing
            # the body. Treat as an error rather than guessing.
            raise RuntimeError(
                "Scan with zero scan-input length and scan outputs is not supported."
            )
        for i, res in enumerate(results):
            seq = res[::-1] if output_directions[i] == 1 else res
            elem_rank = np.asarray(seq[0]).ndim
            norm_out_axis = self._normalize_axis(
                output_axes[i], elem_rank + 1, "scan_output_axes", i
            )
            states.append(np.stack(seq, axis=norm_out_axis))
        return self._check_and_fix_outputs(tuple(states))
