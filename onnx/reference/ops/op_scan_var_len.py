# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto, helper
from onnx.reference.op_run import OpRun


def _resolve_axis(axis: int, rank: int, attr_name: str, index: int) -> int:
    """Normalize a (possibly negative) axis attribute and validate it falls
    within ``[0, rank)``.

    Used for both ``scan_input_axes`` (against each scan input's rank) and
    ``scan_output_axes`` (against each scan output's rank).
    """
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ValueError(
            f"ScanVarLen: {attr_name}[{index}]={axis} is out of range for rank {rank}."
        )
    return normalized


def _normalize_hint(raw_hint, scan_output_index: int) -> np.ndarray | None:
    """Validate and normalize a single shape-hint input.

    Returns a 1-D ``int64`` numpy array, or ``None`` if the hint is absent
    (the model-level ``""`` placeholder, which the reference evaluator
    forwards as Python ``None``). Raises :class:`TypeError` on dtype
    violations and :class:`ValueError` on rank or value violations.

    The dtype check is strict (exactly ``int64``): the op schema requires
    ``tensor(int64)`` for hints, and silently up-casting other integer
    dtypes would mask producer-side bugs that other implementations would
    correctly reject.
    """
    if raw_hint is None:
        return None
    hint_arr = np.asarray(raw_hint)
    if hint_arr.dtype != np.int64:
        raise TypeError(
            f"ScanVarLen scan_output_shape_hint[{scan_output_index}] must be "
            f"tensor(int64), got {hint_arr.dtype}."
        )
    if hint_arr.ndim != 1:
        raise ValueError(
            f"ScanVarLen scan_output_shape_hint[{scan_output_index}] must be a "
            f"1-D tensor; got shape {tuple(hint_arr.shape)}."
        )
    if (hint_arr < 0).any():
        raise ValueError(
            f"ScanVarLen scan_output_shape_hint[{scan_output_index}] contains "
            f"negative values ({hint_arr.tolist()}); all dims must be non-negative."
        )
    return hint_arr


def _body_output_dtype(
    body_runner, body_output_index: int, scan_output_index: int
) -> np.dtype:
    """Return the numpy dtype of body output ``body_output_index`` from its
    declared value_info.

    Used in the zero-iteration path where the body is not executed but the
    scan output's dtype must still be supplied. Raises :class:`ValueError`
    if the body's output value_info lacks a declared element type.
    """
    output_types = getattr(body_runner, "output_types", None)
    if not output_types or body_output_index >= len(output_types):
        raise ValueError(
            f"ScanVarLen: cannot determine dtype for scan output {scan_output_index} "
            f"in the zero-iteration path; the body subgraph has no value_info for "
            f"output {body_output_index}. Supply a body output type annotation or a "
            f"shape hint for this scan output."
        )
    elem_type = output_types[body_output_index].tensor_type.elem_type
    if elem_type == TensorProto.UNDEFINED:
        raise ValueError(
            f"ScanVarLen: body subgraph output {body_output_index} (scan output "
            f"{scan_output_index}) has no declared element type, which is required "
            f"to fabricate a zero-iteration empty output. Add an explicit type "
            f"annotation to the body output, or supply a shape hint for this scan "
            f"output."
        )
    return helper.tensor_dtype_to_np_dtype(elem_type)


def _body_output_static_shape(
    body_runner, body_output_index: int, scan_output_index: int
) -> list[int]:
    """Return the fully-static shape of body output ``body_output_index`` from
    its declared value_info.

    Used in the zero-iteration path without a shape hint to fabricate an
    empty scan output. The caller substitutes ``0`` at the concat axis.
    Raises :class:`ValueError` if the shape is missing or has any symbolic
    dimension.
    """
    type_proto = body_runner.output_types[body_output_index]
    if not type_proto.tensor_type.HasField("shape"):
        raise ValueError(
            f"ScanVarLen: body subgraph output {body_output_index} (scan output "
            f"{scan_output_index}) has no declared shape, which is required to "
            f"fabricate a zero-iteration empty output without a hint. Add a shape "
            f"annotation to the body output, or supply a shape hint for this scan "
            f"output."
        )
    shape: list[int] = []
    for dim_index, dim in enumerate(type_proto.tensor_type.shape.dim):
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            symbol = dim.dim_param or "?"
            raise ValueError(
                f"ScanVarLen: body subgraph output {body_output_index} (scan "
                f"output {scan_output_index}) has a symbolic dimension at index "
                f"{dim_index} ({symbol!r}). The zero-iteration path requires a "
                f"fully-static body output shape, or a shape hint for this scan "
                f"output."
            )
    return shape


class ScanVarLen(OpRun):
    """Reference implementation for the ScanVarLen control-flow operator.

    ScanVarLen generalizes :class:`Scan`: each iteration's contribution to a
    scan output may be of variable size along ``scan_output_axes[i]``. After
    the loop, per-iteration contributions are concatenated (not stacked)
    along that axis.

    Inputs (single trailing variadic, in order):
      * ``N`` initial loop-state values
      * ``M`` scan inputs (``M == num_scan_inputs``)
      * Optionally ``K`` shape hints (one per scan output, in scan-output
        order). Each hint is a 1-D ``int64`` tensor giving the full expected
        shape of the corresponding scan output (concat-axis entry is the
        declared total). Pass ``""`` (which the reference evaluator forwards
        as Python ``None``) for any individual scan output without a hint.
        Either omit the hint group entirely (no hints at all) or supply
        exactly ``K`` slots; any other count is a schema error.

    ``N`` is derived from the body subgraph:
    ``N = body.input_count - num_scan_inputs``. ``K`` is similarly derived:
    ``K = body.output_count - N``.

    Outputs: ``N`` final loop-state values followed by ``K`` concatenated
    scan outputs.

    Notes:
        * Heterogeneous variadic inputs/outputs.
        * Supports per-input reverse iteration via ``scan_input_directions``.
        * Supports non-zero (including negative) ``scan_input_axes`` /
          ``scan_output_axes``.
        * Zero-iteration scans are defined behavior. When the scan-input
          sequence axis has size 0, the body is not run and each scan output
          is returned as an empty tensor:

          - If a shape hint is supplied for scan output ``i``, the output
            shape is ``hint`` with ``hint[scan_output_axes[i]]`` replaced by
            ``0``. The dtype matches the body subgraph's declared output
            type.
          - If no hint is supplied, the output shape is read from the body
            subgraph's declared output value_info (with the concat-axis
            entry replaced by ``0``); the body output type annotation must
            be fully static for this path to succeed.
        * When a hint is present and the loop runs at least once, the
          concatenated scan output's shape is validated against the hint
          along every axis (including the concat axis).
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
        body=None,  # noqa: ARG002  # The bound subgraph runner lives on self.body.
        num_scan_inputs=None,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        attributes=None,  # noqa: ARG002
    ):
        # self.body is the bound subgraph runner; the homonymous kwarg above
        # (the raw GraphProto attribute) is unused here.
        body_runner = self.body
        num_scan_inputs_value = (
            num_scan_inputs if num_scan_inputs is not None else self.num_scan_inputs
        )
        if num_scan_inputs_value is None or num_scan_inputs_value < 1:
            raise RuntimeError(
                "ScanVarLen requires at least one scan input (num_scan_inputs >= 1)."
            )

        # Derive N and K from the body subgraph (the spec ground truth for both).
        body_input_count = len(body_runner.input_names)
        if body_input_count < num_scan_inputs_value:
            raise ValueError(
                f"ScanVarLen body subgraph has {body_input_count} input(s) but expected "
                f"at least num_scan_inputs={num_scan_inputs_value}."
            )
        num_loop_state_vars = body_input_count - num_scan_inputs_value
        body_output_count = len(body_runner.output_names)
        if body_output_count < num_loop_state_vars:
            raise ValueError(
                f"ScanVarLen body subgraph has {body_output_count} output(s) but "
                f"expected at least {num_loop_state_vars} (one per loop state variable)."
            )
        num_scan_outputs = body_output_count - num_loop_state_vars

        # Split the trailing variadic into [N state][M scan_inputs][hints].
        min_required = num_loop_state_vars + num_scan_inputs_value
        if len(args) < min_required:
            raise RuntimeError(
                f"ScanVarLen: expected at least {min_required} variadic inputs "
                f"({num_loop_state_vars} state vars + {num_scan_inputs_value} scan "
                f"inputs), got {len(args)}."
            )
        states = list(args[:num_loop_state_vars])
        scan_values = list(args[num_loop_state_vars:min_required])
        hint_args = list(args[min_required:])

        # Hint count must be exactly 0 or K. Any partial count is a schema error.
        if hint_args and len(hint_args) != num_scan_outputs:
            raise ValueError(
                f"ScanVarLen: shape hint count ({len(hint_args)}) must equal scan "
                f"output count ({num_scan_outputs}), or zero hints may be supplied "
                f"for the no-hint case."
            )
        if not hint_args:
            hints: list[np.ndarray | None] = [None] * num_scan_outputs
        else:
            hints = [_normalize_hint(h, i) for i, h in enumerate(hint_args)]

        # Resolve per-input axes/directions; normalize and validate negative axes
        # against each scan input's rank.
        input_axes = [
            (
                0
                if scan_input_axes is None or i >= len(scan_input_axes)
                else int(scan_input_axes[i])
            )
            for i in range(num_scan_inputs_value)
        ]
        for i, axis in enumerate(input_axes):
            input_axes[i] = _resolve_axis(
                axis, scan_values[i].ndim, "scan_input_axes", i
            )
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
        # Output axes are resolved+validated against each scan output's rank
        # below: in the zero-iter path against the hint or body value_info rank;
        # in the main-loop path against the first iteration's chunk rank.
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

        # ── Zero-iteration path ──────────────────────────────────────────────
        # Defined behavior: the body does not run; scan outputs are empty tensors
        # whose non-concat dims come from the hint (when present) or the body's
        # declared output value_info (otherwise), and whose concat-axis is 0.
        if sequence_length == 0:
            scan_outputs: list[np.ndarray] = []
            for i in range(num_scan_outputs):
                body_out_idx = num_loop_state_vars + i
                dtype = _body_output_dtype(body_runner, body_out_idx, i)
                hint = hints[i]
                if hint is not None:
                    shape = [int(x) for x in hint.tolist()]
                else:
                    shape = _body_output_static_shape(body_runner, body_out_idx, i)
                axis = _resolve_axis(
                    output_axes_attr[i], len(shape), "scan_output_axes", i
                )
                shape[axis] = 0
                scan_outputs.append(np.empty(shape, dtype=dtype))
            return self._check_and_fix_outputs(tuple(states + scan_outputs))

        # ── Main loop (sequence_length >= 1) ─────────────────────────────────
        state_names_in = body_runner.input_names[:num_loop_state_vars]
        scan_names_in = body_runner.input_names[num_loop_state_vars:]
        state_names_out = body_runner.output_names[:num_loop_state_vars]
        scan_names_out = body_runner.output_names[num_loop_state_vars:]

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
            outputs = dict(zip(body_runner.output_names, outputs_list, strict=False))
            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                per_iter_outputs[i].append(outputs[name])

        # Concatenate per-iteration scan outputs and validate against hints.
        final_scan_outputs: list[np.ndarray] = []
        for i in range(num_scan_outputs):
            chunks = per_iter_outputs[i]
            # The ONNX schema requires consistent ranks across iterations,
            # so normalizing against the first chunk's rank is sufficient.
            axis = _resolve_axis(
                output_axes_attr[i], chunks[0].ndim, "scan_output_axes", i
            )
            # Defensive: verify rank and non-concat dims are consistent
            # across iterations before concatenating. np.concatenate would
            # raise on mismatch but with a less specific message.
            base_rank = chunks[0].ndim
            base_other_dims = tuple(
                d for k, d in enumerate(chunks[0].shape) if k != axis
            )
            for t, chunk in enumerate(chunks[1:], start=1):
                if chunk.ndim != base_rank:
                    raise ValueError(
                        f"ScanVarLen: scan output {i} has inconsistent ranks "
                        f"across iterations (iteration 0 produced rank "
                        f"{base_rank}, iteration {t} produced rank "
                        f"{chunk.ndim})."
                    )
                other_dims = tuple(d for k, d in enumerate(chunk.shape) if k != axis)
                if other_dims != base_other_dims:
                    raise ValueError(
                        f"ScanVarLen: scan output {i} has inconsistent non-concat "
                        f"dimensions across iterations (iteration 0 shape "
                        f"{tuple(chunks[0].shape)}, iteration {t} shape "
                        f"{tuple(chunk.shape)}, concat axis {axis})."
                    )
            result = np.concatenate(chunks, axis=axis)

            # Runtime consistency check: full-shape match against the hint
            # (every axis, including the concat axis).
            hint = hints[i]
            if hint is not None:
                if hint.shape[0] != result.ndim:
                    raise ValueError(
                        f"ScanVarLen: shape hint for scan output {i} has length "
                        f"{hint.shape[0]} but scan output {i} has rank "
                        f"{result.ndim}."
                    )
                for j, hint_dim in enumerate(hint.tolist()):
                    hint_dim_value = int(hint_dim)
                    actual_dim = int(result.shape[j])
                    if hint_dim_value != actual_dim:
                        which = "concat axis" if j == axis else "non-concat axis"
                        raise ValueError(
                            f"ScanVarLen: shape hint for scan output {i}[{j}]="
                            f"{hint_dim_value} does not match scan output {i} "
                            f"actual dim {actual_dim} ({which}={axis})."
                        )
            final_scan_outputs.append(result)

        return self._check_and_fix_outputs(tuple(states + final_scan_outputs))
