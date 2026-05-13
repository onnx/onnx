# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _resolve_output_axis(axis: int, rank: int, output_index: int) -> int:
    """Normalize a (possibly negative) ``scan_output_axes`` value and validate
    it falls within ``[0, rank)`` for the corresponding scan output.
    """
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ValueError(
            f"ScanVarLen: scan_output_axes[{output_index}]={axis} is out of range "
            f"for scan output {output_index} of rank {rank}."
        )
    return normalized


class ScanVarLen(OpRun):
    """Reference implementation for the ScanVarLen control-flow operator.

    ScanVarLen generalizes :class:`Scan`: each iteration's contribution to a
    scan output may be of variable size along ``scan_output_axes[i]``. After
    the loop, per-iteration contributions are concatenated (not stacked) along
    that axis.

    Inputs (positional, in order):
      * ``output_lengths`` — optional ``int64[K]`` tensor. When supplied, the
        op enforces that ``output_lengths[i]`` equals the total concat-axis
        size of scan output ``i``; a mismatch raises :class:`ValueError`.
      * ``initial_state`` (``N`` tensors) followed by ``scan_inputs``
        (``num_scan_inputs`` tensors), as in :class:`Scan`.

    Outputs: ``N`` final state values followed by ``K`` concatenated scan
    outputs.

    Notes:
        * Heterogeneous variadic inputs/outputs.
        * Supports per-input reverse iteration via ``scan_input_directions``.
        * Supports non-zero (including negative) ``scan_input_axes`` /
          ``scan_output_axes``.
        * Zero-iteration scans produce shape-correct empty outputs by
          performing a single "dry run" of the body subgraph using
          zero-filled per-iteration inputs derived from the scan-input
          shapes; the dimension at ``scan_output_axes[i]`` is then
          collapsed to ``0``.
    """

    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        if not hasattr(self.body, "run"):
            raise RuntimeError(
                f"Parameter 'body' must have a method 'run', type {type(self.body)}."
            )

    def _run(  # type: ignore[override]
        self,
        *args,
        body=None,
        num_scan_inputs=None,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        attributes=None,  # noqa: ARG002
    ):
        del body  # The subgraph is accessed via ``self.body``/``self._run_body``.
        if not args:
            raise RuntimeError(
                "ScanVarLen requires at least the optional 'output_lengths' input slot "
                "followed by state and scan inputs."
            )

        # First positional input is the optional output_lengths (may be None).
        output_lengths = args[0]
        rest = args[1:]

        num_scan_inputs_value = (
            num_scan_inputs if num_scan_inputs is not None else self.num_scan_inputs
        )
        num_loop_state_vars = len(rest) - num_scan_inputs_value
        if num_loop_state_vars < 0:
            raise RuntimeError(
                f"ScanVarLen: number of variadic inputs ({len(rest)}) is less than "
                f"num_scan_inputs ({num_scan_inputs_value})."
            )

        states = list(rest[:num_loop_state_vars])
        scan_values = list(rest[num_loop_state_vars:])

        body = self.body
        state_names_in = body.input_names[:num_loop_state_vars]
        scan_names_in = body.input_names[num_loop_state_vars:]
        state_names_out = body.output_names[:num_loop_state_vars]
        scan_names_out = body.output_names[num_loop_state_vars:]
        num_scan_outputs = len(scan_names_out)
        expected_body_inputs = num_loop_state_vars + num_scan_inputs_value
        if len(body.input_names) != expected_body_inputs:
            raise ValueError(
                f"ScanVarLen body subgraph has {len(body.input_names)} input(s) but "
                f"expected {expected_body_inputs} "
                f"(num_loop_state_vars={num_loop_state_vars} + "
                f"num_scan_inputs={num_scan_inputs_value})."
            )
        if len(body.output_names) < num_loop_state_vars:
            raise ValueError(
                f"ScanVarLen body subgraph has {len(body.output_names)} output(s) but "
                f"expected at least {num_loop_state_vars} (one per loop state variable)."
            )

        # Resolve per-input axes/directions; normalize and validate negative axes
        # against each scan input's rank.
        if num_scan_inputs_value == 0:
            raise RuntimeError(
                "ScanVarLen requires at least one scan input (num_scan_inputs >= 1)."
            )
        input_axes = [
            (
                0
                if scan_input_axes is None or i >= len(scan_input_axes)
                else int(scan_input_axes[i])
            )
            for i in range(num_scan_inputs_value)
        ]
        for i, axis in enumerate(input_axes):
            rank = scan_values[i].ndim
            normalized = axis + rank if axis < 0 else axis
            if not 0 <= normalized < rank:
                raise ValueError(
                    f"ScanVarLen: scan_input_axes[{i}]={axis} is out of range for "
                    f"scan input {i} of rank {rank}."
                )
            input_axes[i] = normalized
        input_directions = [
            (
                0
                if scan_input_directions is None or i >= len(scan_input_directions)
                else int(scan_input_directions[i])
            )
            for i in range(num_scan_inputs_value)
        ]
        for i, direction in enumerate(input_directions):
            if direction not in (0, 1):
                raise ValueError(
                    f"ScanVarLen: scan_input_directions[{i}]={direction} is invalid; "
                    f"expected 0 (forward) or 1 (reverse)."
                )
        # Output axes are resolved+validated against each scan output's rank later
        # (after we have a sample output array).
        output_axes_attr = [
            (
                0
                if scan_output_axes is None or i >= len(scan_output_axes)
                else int(scan_output_axes[i])
            )
            for i in range(num_scan_outputs)
        ]

        # Sequence length comes from the first scan input along its sequence axis;
        # validate that all scan inputs agree on the sequence length.
        sequence_length = int(scan_values[0].shape[input_axes[0]])
        for i in range(1, num_scan_inputs_value):
            length_i = int(scan_values[i].shape[input_axes[i]])
            if length_i != sequence_length:
                raise ValueError(
                    f"ScanVarLen: scan input {i} has sequence length {length_i} along "
                    f"axis {input_axes[i]}, but expected {sequence_length} (from scan "
                    f"input 0 along axis {input_axes[0]})."
                )

        per_iter_outputs: list[list[np.ndarray]] = [[] for _ in range(num_scan_outputs)]
        for t in range(sequence_length):
            inputs = dict(zip(state_names_in, states, strict=False))
            for i, name in enumerate(scan_names_in):
                idx = (sequence_length - 1 - t) if input_directions[i] == 1 else t
                inputs[name] = np.take(scan_values[i], idx, axis=input_axes[i])
            try:
                outputs_list = self._run_body(inputs)
            except TypeError as e:
                raise TypeError(
                    f"Unable to call 'run' for type '{type(self.body)}'."
                ) from e
            outputs = dict(zip(body.output_names, outputs_list, strict=False))
            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                per_iter_outputs[i].append(outputs[name])

        final_scan_outputs: list[np.ndarray] = []
        # Resolve and validate output axes against the rank of a representative
        # scan output (either from the first iteration or from a dry run).
        resolved_output_axes: list[int] = []
        if sequence_length == 0 and num_scan_outputs > 0:
            # Dry-run the body once with zero-filled per-iteration inputs to
            # learn the non-concat dims of each scan output.
            dry_inputs = dict(zip(state_names_in, states, strict=False))
            for i, name in enumerate(scan_names_in):
                scan_value = scan_values[i]
                axis = input_axes[i]
                slice_shape = scan_value.shape[:axis] + scan_value.shape[axis + 1 :]
                dry_inputs[name] = np.zeros(slice_shape, dtype=scan_value.dtype)
            try:
                dry_outputs_list = self._run_body(dry_inputs)
            except TypeError as e:
                raise TypeError(
                    f"Unable to call 'run' for type '{type(self.body)}'."
                ) from e
            dry_outputs = dict(zip(body.output_names, dry_outputs_list, strict=False))
            for i, name in enumerate(scan_names_out):
                prototype = np.asarray(dry_outputs[name])
                axis = _resolve_output_axis(output_axes_attr[i], prototype.ndim, i)
                resolved_output_axes.append(axis)
                shape = list(prototype.shape)
                shape[axis] = 0
                final_scan_outputs.append(np.zeros(tuple(shape), dtype=prototype.dtype))
        else:
            for i in range(num_scan_outputs):
                chunks = per_iter_outputs[i]
                # The ONNX schema requires consistent ranks across iterations,
                # so normalizing against the first chunk's rank is sufficient.
                axis = _resolve_output_axis(output_axes_attr[i], chunks[0].ndim, i)
                resolved_output_axes.append(axis)
                final_scan_outputs.append(np.concatenate(chunks, axis=axis))

        if output_lengths is not None:
            output_lengths_arr = np.asarray(output_lengths)
            if not np.issubdtype(output_lengths_arr.dtype, np.integer):
                raise ValueError(
                    f"ScanVarLen: 'output_lengths' must be an integer tensor "
                    f"(tensor(int64) per the op schema); got dtype "
                    f"{output_lengths_arr.dtype}."
                )
            if (
                output_lengths_arr.ndim != 1
                or output_lengths_arr.shape[0] != num_scan_outputs
            ):
                raise ValueError(
                    f"ScanVarLen: 'output_lengths' must be a 1-D tensor of length "
                    f"{num_scan_outputs} (number of scan outputs), got shape "
                    f"{tuple(output_lengths_arr.shape)}."
                )
            for i, arr in enumerate(final_scan_outputs):
                axis = resolved_output_axes[i]
                expected = int(output_lengths_arr[i])
                actual = int(arr.shape[axis])
                if expected != actual:
                    raise ValueError(
                        f"ScanVarLen: output_lengths[{i}]={expected} does not match "
                        f"the actual concat-axis size {actual} along axis {axis} "
                        f"of scan output {i}."
                    )

        return self._check_and_fix_outputs(tuple(states + final_scan_outputs))
