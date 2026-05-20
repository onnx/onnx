# Scan: enable lexical capture from enclosing graph

Changes to `onnx/reference/ops/op_scan.py` to make the reference `Scan`
implementation honor ONNX's lexical-capture semantics: the body subgraph
may reference names defined in the enclosing graph, and those names must
resolve at body-evaluation time.

## Summary

1. Override `need_context()` to return `True` so the runtime passes the
   outer-scope `context` into `_run`.
2. Accept a `context=None` keyword in `_run`.
3. Seed each iteration's `inputs` dict with the outer-scope `context`
   first, then overlay per-iteration loop-state and scan-slice inputs so
   that same-named locals correctly shadow outer values.

## Diff

```diff
--- a/onnx/reference/ops/op_scan.py
+++ b/onnx/reference/ops/op_scan.py
@@ -43,6 +43,13 @@ class Scan(OpRun):
         self.input_names = self.body.input_names
         self.output_names = self.body.output_names
 
+    def need_context(self) -> bool:
+        """Scan body subgraphs may reference names from the enclosing graph
+        (lexical capture, per the ONNX spec). Request the outer-scope
+        context so those names resolve at body-evaluation time.
+        """
+        return True
+
     def _common_run_shape(self, *args):
         num_loop_state_vars = len(args) - self.num_scan_inputs
         num_scan_outputs = len(args) - num_loop_state_vars
@@ -99,6 +106,7 @@ class Scan(OpRun):
     def _run(  # type: ignore[override]
         self,
         *args,
+        context=None,
         body=None,  # noqa: ARG002
         num_scan_inputs=None,  # noqa: ARG002
         scan_input_axes=None,  # noqa: ARG002
@@ -127,7 +135,13 @@ class Scan(OpRun):
         results = [[] for _ in scan_names_out]
 
         for it in range(max_iter):
-            inputs = dict(zip(state_names_in, states, strict=False))
+            # Seed with outer-scope values first; per-iteration state and
+            # scan-slice inputs (added below) shadow any same-named outer
+            # values, matching ONNX's lexical-capture semantics.
+            inputs: dict = {}
+            if context is not None:
+                inputs.update(context)
+            inputs.update(dict(zip(state_names_in, states, strict=False)))
             inputs.update(
                 {
                     name: value[it]
```

## Notes for the follow-up PR

- This mirrors the pattern used by other subgraph-bearing ops (e.g.
  `If`, `Loop`) that already implement `need_context()`.
- Shadowing order matters: outer context first, then loop-state, then
  scan-slice inputs — locals must win over captured names.
- Add a regression test where a `Scan` body references a tensor defined
  in the parent graph (not listed among the body's formal inputs).
