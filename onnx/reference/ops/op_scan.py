# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Scan(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        if not hasattr(self.body, "run"):
            raise RuntimeError(
                f"Parameter 'body' must have a method 'run', type {type(self.body)}."
            )
        if self.num_scan_inputs <= 0:
            raise RuntimeError(
                f"Scan requires num_scan_inputs > 0, got {self.num_scan_inputs}."
            )
        self.input_directions_ = [
            (
                0
                if self.scan_input_directions is None
                or i >= len(self.scan_input_directions)
                else self.scan_input_directions[i]
            )
            for i in range(self.num_scan_inputs)
        ]
        if self.input_directions_:
            max_dir_in = max(self.input_directions_)
            if max_dir_in != 0:
                raise RuntimeError(
                    "Scan is not implemented for other output input_direction than 0."
                )
        self.input_axes_ = [
            (
                0
                if self.scan_input_axes is None or i >= len(self.scan_input_axes)
                else self.scan_input_axes[i]
            )
            for i in range(self.num_scan_inputs)
        ]
        if self.input_axes_:
            max_axe_in = max(self.input_axes_)
            if max_axe_in != 0:
                raise RuntimeError(
                    "Scan is not implemented for other input axes than 0."
                )
        self.input_names = self.body.input_names
        self.output_names = self.body.output_names

    def _common_run_shape(self, *args):
        num_loop_state_vars = len(args) - self.num_scan_inputs
        num_scan_outputs = len(self.output_names) - num_loop_state_vars

        output_directions = [
            (
                0
                if self.scan_output_directions is None
                or i >= len(self.scan_output_directions)
                else self.scan_output_directions[i]
            )
            for i in range(num_scan_outputs)
        ]
        max_dir_out = max(output_directions) if output_directions else 0
        if max_dir_out != 0:
            raise RuntimeError(
                "Scan is not implemented for other output output_direction than 0."
            )
        output_axes = [
            (
                0
                if self.scan_output_axes is None or i >= len(self.scan_output_axes)
                else self.scan_output_axes[i]
            )
            for i in range(num_scan_outputs)
        ]
        max_axe_out = max(output_axes) if output_axes else 0
        if max_axe_out != 0:
            raise RuntimeError("Scan is not implemented for other output axes than 0.")

        state_names_in = self.input_names[:num_loop_state_vars]
        state_names_out = self.output_names[: len(state_names_in)]
        scan_names_in = self.input_names[num_loop_state_vars:]
        scan_names_out = self.output_names[num_loop_state_vars:]
        scan_values = args[num_loop_state_vars:]

        states = list(args[:num_loop_state_vars])

        return (
            num_loop_state_vars,
            num_scan_outputs,
            output_directions,
            max_dir_out,
            output_axes,
            max_axe_out,
            state_names_in,
            state_names_out,
            scan_names_in,
            scan_names_out,
            scan_values,
            states,
        )

    def _run(  # type: ignore[override]
        self,
        *args,
        body=None,  # noqa: ARG002
        num_scan_inputs=None,  # noqa: ARG002
        scan_input_axes=None,  # noqa: ARG002
        scan_input_directions=None,  # noqa: ARG002
        scan_output_axes=None,  # noqa: ARG002
        scan_output_directions=None,  # noqa: ARG002
        attributes=None,  # noqa: ARG002
    ):
        # TODO: support overridden attributes.
        (
            num_loop_state_vars,
            _num_scan_outputs,
            _output_directions,
            _max_dir_out,
            _output_axes,
            _max_axe_out,
            state_names_in,
            state_names_out,
            scan_names_in,
            scan_names_out,
            scan_values,
            states,
        ) = self._common_run_shape(*args)

        max_iter = args[num_loop_state_vars].shape[self.input_axes_[0]]
        results = [[] for _ in scan_names_out]

        for it in range(max_iter):
            inputs = dict(zip(state_names_in, states, strict=False))
            inputs.update(
                {
                    name: value[it]
                    for name, value in zip(scan_names_in, scan_values, strict=False)
                }
            )

            try:
                outputs_list = self._run_body(inputs)
            except TypeError as e:
                raise TypeError(
                    f"Unable to call 'run' for type '{type(self.body)}'."
                ) from e

            outputs = dict(zip(self.output_names, outputs_list, strict=False))
            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                results[i].append(np.expand_dims(outputs[name], axis=0))

        for i, res in enumerate(results):
            if res:
                states.append(np.concatenate(res, axis=0))
            else:
                states.append(
                    self._empty_scan_output(num_loop_state_vars + i, scan_values, i)
                )
        return self._check_and_fix_outputs(tuple(states))

    def _empty_scan_output(self, body_output_index, scan_values, scan_output_index):
        """Build a zero-length scan output preserving the per-iteration shape.

        Uses ``self.body.output_types_`` (populated by the reference evaluator)
        as the source of truth for the trailing element shape and dtype. Unknown
        or symbolic dims map to 0 so ``np.zeros`` is safe. Falls back to the
        previous ``(0,)`` shape only when no type info is available at all.
        """
        output_types = getattr(self.body, "output_types_", None)
        type_proto = (
            output_types[body_output_index]
            if output_types is not None and body_output_index < len(output_types)
            else None
        )

        # Resolve dtype, preferring the body's declared element type.
        dtype = None
        if type_proto is not None and type_proto.tensor_type.elem_type:
            from onnx.helper import tensor_dtype_to_np_dtype

            dtype = tensor_dtype_to_np_dtype(type_proto.tensor_type.elem_type)
        if dtype is None:
            if scan_values:
                dtype = scan_values[
                    min(scan_output_index, len(scan_values) - 1)
                ].dtype
            else:
                dtype = np.float32

        # Resolve trailing shape from the body output, when present.
        if type_proto is not None and type_proto.tensor_type.HasField("shape"):
            trailing_dims = tuple(
                d.dim_value if d.HasField("dim_value") and d.dim_value >= 0 else 0
                for d in type_proto.tensor_type.shape.dim
            )
            return np.zeros((0, *trailing_dims), dtype=dtype)
        return np.empty((0,), dtype=dtype)
