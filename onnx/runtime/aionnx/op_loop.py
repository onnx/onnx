# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0914,W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Loop(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if "opsets" not in self.run_params:
            raise KeyError("run_params must contains key 'opsets'.")
        if "verbose" not in run_params:
            raise KeyError("run_params must contains key 'verbose'.")

    def need_context(self) -> bool:
        """
        The operator Loop needs to know all results produced
        so far as the loop may silently access one of them.
        Some information are not always referred in the list of inputs
        (kind of static variables).
        """
        return True

    def _run(self, M, cond, *args, context=None):  # type: ignore
        if len(args) > 0:
            v_initial = args[0]
            args = args[1:]
        else:
            v_initial = None
        body = self.body  # type: ignore
        loop_inputs = body.input_names
        inputs = {name: None for name in loop_inputs}
        if v_initial is not None:
            inputs[loop_inputs[2]] = v_initial
        cond_name = body.output_names[0]
        if len(args) > 0:
            begin = len(loop_inputs) - len(args)
            all_inputs = loop_inputs[begin:]
            for name, val in zip(all_inputs, args):
                inputs[name] = val
        if context is not None:
            for a in context:
                inputs[a] = context[a]

        it = 0
        while cond and it < M:
            self._log("  -- loop> {%r}", context)
            if len(body.input_names) > 0 and body.input_names[0] is not None:
                inputs[body.input_names[0]] = numpy.array(it, dtype=M.dtype)
            if len(body.input_names) > 1 and body.input_names[1] is not None:
                inputs[body.input_names[1]] = cond
            outputs = self._run_body(inputs)  # type: ignore
            cond = outputs[cond_name]
            if cond is None:
                raise RuntimeError(
                    f"Condition {cond_name!r} returned by the subgraph cannot be None."
                )
            for i, o in zip(body.input_names[2:], body.output_names[1:]):
                inputs[i] = outputs[o]
            it += 1
            self._log("  -- loop<")

        if it == 0:
            outputs = {body.output_names[1]: cond}
            for i, o in zip(body.input_names[2:], body.output_names[1:]):
                outputs[o] = inputs[i]
        for o in body.output_names:
            if o not in outputs:
                outputs[o] = numpy.empty(shape=tuple())
        res = tuple(body.output_names[1:])
        if any(r is None for r in res):
            raise TypeError("Operator Loop produces a None value.")
        return res
