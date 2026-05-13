# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Node test cases for the ``ScanVarLen`` control-flow operator.

ScanVarLen is a generalization of :class:`Scan` introduced in ai.onnx opset
27. Each iteration's contribution to a scan output may have a variable size
along ``scan_output_axes[i]``; the per-iteration outputs are concatenated
(not stacked) along that axis after the loop.

Body subgraphs are expressed using :func:`onnx.parser.parse_graph` for
readability; the outer ``ScanVarLen`` node is built with the standard
``helper.make_node`` + :func:`expect` pattern used elsewhere in node tests.
"""

from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

_OPSET_IMPORTS = [onnx.helper.make_opsetid("", 27)]


def _identity_body(body_name: str) -> onnx.GraphProto:
    """Body subgraph with a single length-4 scan input passed through
    unchanged as a single scan output; no loop-state variables.
    """
    return onnx.parser.parse_graph(
        f"""
        {body_name} (float[4] scan_in) => (float[4] scan_out)
        {{
            scan_out = Identity(scan_in)
        }}
        """
    )


class ScanVarLen(Base):
    @staticmethod
    def export_scan_var_len_basic() -> None:
        """Single scan input, single scan output, default axes/direction, no
        ``output_lengths``. Verifies the concat (not stack) semantics: three
        iterations each contributing a length-4 slice yield a length-12
        output along the default concat axis 0.
        """
        body = _identity_body("scan_var_len_basic_body")
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["", "scan_input"],  # "" = omitted optional output_lengths
            outputs=["scan_output"],
            num_scan_inputs=1,
            body=body,
        )
        scan_input = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Iterations contribute rows 0, 1, 2; concat along axis 0 yields the
        # flattened sequence [0..11].
        scan_output = scan_input.reshape(12)

        expect(
            node,
            inputs=[scan_input],
            outputs=[scan_output],
            name="test_scan_var_len_basic",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_scan_var_len_output_lengths() -> None:
        """Same as the basic variant but with an explicit ``output_lengths``
        input. The supplied length must equal the total concat-axis size of
        the scan output.
        """
        body = _identity_body("scan_var_len_output_lengths_body")
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["output_lengths", "scan_input"],
            outputs=["scan_output"],
            num_scan_inputs=1,
            body=body,
        )
        output_lengths = np.array([12], dtype=np.int64)
        scan_input = np.arange(12, dtype=np.float32).reshape(3, 4)
        scan_output = scan_input.reshape(12)

        expect(
            node,
            inputs=[output_lengths, scan_input],
            outputs=[scan_output],
            name="test_scan_var_len_output_lengths",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_scan_var_len_reverse() -> None:
        """``scan_input_directions=[1]`` makes the (only) scan input iterate
        from last to first; per-iteration outputs are still concatenated in
        iteration order, so the final concat output is the reversed input
        along axis 0.
        """
        body = _identity_body("scan_var_len_reverse_body")
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["", "scan_input"],
            outputs=["scan_output"],
            num_scan_inputs=1,
            scan_input_directions=[1],
            body=body,
        )
        scan_input = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Reverse-iteration: rows are visited 2, 1, 0; concatenated along
        # axis 0 yields the row-reversed input flattened.
        scan_output = scan_input[::-1].reshape(12)

        expect(
            node,
            inputs=[scan_input],
            outputs=[scan_output],
            name="test_scan_var_len_reverse",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_scan_var_len_axes() -> None:
        """Non-default ``scan_input_axes=[1]`` and ``scan_output_axes=[1]``.

        The scan input is sliced along axis 1 (sequence axis = 1, length 4).
        The body unsqueezes the per-iteration slice so it carries the
        concat dimension explicitly; concatenation along axis 1 across all
        iterations reconstructs the original input.
        """
        # Body: input shape [3] (axis-1 slice of a [3, 4] tensor); output
        # shape [3, 1] (unsqueeze a length-1 concat axis at position 1).
        body = onnx.parser.parse_graph(
            """
            scan_var_len_axes_body (float[3] scan_in) => (float[3, 1] scan_out)
            <int64[1] unsqueeze_axes = {1}>
            {
                scan_out = Unsqueeze(scan_in, unsqueeze_axes)
            }
            """
        )
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["", "scan_input"],
            outputs=["scan_output"],
            num_scan_inputs=1,
            scan_input_axes=[1],
            scan_output_axes=[1],
            body=body,
        )
        scan_input = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Per-iter slice along axis 1 is a length-3 column; unsqueezed to
        # [3, 1] and concatenated along axis 1 over 4 iterations
        # reconstructs the original [3, 4] tensor.
        scan_output = scan_input.copy()

        expect(
            node,
            inputs=[scan_input],
            outputs=[scan_output],
            name="test_scan_var_len_axes",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_scan_var_len_state() -> None:
        """One loop-state variable combined with one scan input and one scan
        output. Mirrors the classic Scan running-sum example but with the
        ScanVarLen concat semantics (each iter contributes a length-2
        slice, concatenated to a length-6 output).
        """
        body = onnx.parser.parse_graph(
            """
            scan_var_len_state_body
                (float[2] state_in, float[2] scan_in)
                => (float[2] state_out, float[2] scan_out)
            {
                state_out = Add(state_in, scan_in)
                scan_out = Identity(state_out)
            }
            """
        )
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["", "initial_state", "scan_input"],
            outputs=["final_state", "scan_output"],
            num_scan_inputs=1,
            body=body,
        )
        initial_state = np.zeros((2,), dtype=np.float32)
        scan_input = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        # Running sum at each step: [1,2], [4,6], [9,12]; final state = [9,12];
        # scan_output is the concatenation of the three running sums along
        # axis 0, yielding a length-6 vector.
        final_state = np.array([9, 12], dtype=np.float32)
        scan_output = np.array([1, 2, 4, 6, 9, 12], dtype=np.float32)

        expect(
            node,
            inputs=[initial_state, scan_input],
            outputs=[final_state, scan_output],
            name="test_scan_var_len_state",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_scan_var_len_ragged() -> None:
        """Body produces variable-sized per-iteration outputs along the concat
        axis. This is the test that actually exercises the 'VarLen' part:
        each iteration slices a different prefix length of its per-iter data
        input, driven by a second scan input that supplies the length.

        The final concatenated scan output has length 1 + 2 + 3 = 6, which is
        also asserted via the optional ``output_lengths`` input.
        """
        # Body inputs: data of shape [4] (fixed) and length of shape [1]
        # (int64). Body output: Slice(data, [0], length, [0]) — a 1-D tensor
        # of dynamic length (``float[?]``).
        body = onnx.parser.parse_graph(
            """
            scan_var_len_ragged_body
                (float[4] data_in, int64[1] length_in)
                => (float[?] scan_out)
            <int64[1] starts = {0}, int64[1] slice_axes = {0}>
            {
                scan_out = Slice(data_in, starts, length_in, slice_axes)
            }
            """
        )
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["output_lengths", "data", "lengths"],
            outputs=["scan_output"],
            num_scan_inputs=2,
            body=body,
        )
        output_lengths = np.array([6], dtype=np.int64)
        data = np.array(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
            dtype=np.float32,
        )
        lengths = np.array([[1], [2], [3]], dtype=np.int64)
        # Iter t slices data[t, 0:lengths[t]]; concat along axis 0 yields
        # [10] + [20, 21] + [30, 31, 32] = length 6.
        scan_output = np.array([10, 20, 21, 30, 31, 32], dtype=np.float32)

        expect(
            node,
            inputs=[output_lengths, data, lengths],
            outputs=[scan_output],
            name="test_scan_var_len_ragged",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_scan_var_len_zero_iterations() -> None:
        """Scan input has sequence_length=0 along the sequence axis. The ref
        impl must do a body 'dry run' to recover the non-concat dims and
        emit a shape-correct empty output (size 0 along the concat axis).
        """
        body = _identity_body("scan_var_len_zero_iterations_body")
        node = onnx.helper.make_node(
            "ScanVarLen",
            inputs=["", "scan_input"],
            outputs=["scan_output"],
            num_scan_inputs=1,
            body=body,
        )
        # Zero iterations: sequence axis (default 0) has length 0.
        scan_input = np.zeros((0, 4), dtype=np.float32)
        # With identity body producing per-iter shape [4] and default
        # scan_output_axes=[0], 0 iterations yield a length-0 1-D output.
        scan_output = np.zeros((0,), dtype=np.float32)

        expect(
            node,
            inputs=[scan_input],
            outputs=[scan_output],
            name="test_scan_var_len_zero_iterations",
            opset_imports=_OPSET_IMPORTS,
        )
