# Adding a Function Body Definition for an Operator

Canonical guide: [`.agents/skills/add-function-body/SKILL.md`](../../.agents/skills/add-function-body/SKILL.md). Background: [`docs/AddFunctionBody.md`](../../docs/AddFunctionBody.md).

## Critical reminders

- Function body lives inline in the schema in `onnx/defs/<domain>/defs.cc` via `.FunctionBody(R"ONNX(...)")` (simple) or `.SetContextDependentFunctionBodyBuilder(...)` (context-dependent).
- For context-dependent builders, always finalize with `schema.BuildFunction(functionProto)` and `return true`.
- Variable names in the body must not collide with declared input/output names. Use `CastLike` (not `Cast`) when the target dtype depends on another input. Reference enclosing-op attributes with `@attr_name` — only for attributes declared in `.Attr(...)` calls.
- The body must produce all declared outputs.
- Prefer named `static bool` builder functions over inline lambdas (macro expansion breaks debugger breakpoints). Simple string-based `.FunctionBody(R"ONNX(...)")` is fine as-is.
- For the ONNX text format itself (syntax, `Constant <value = ...>`, body subgraphs, parser tests), see [`.agents/skills/onnxtxt/SKILL.md`](../../.agents/skills/onnxtxt/SKILL.md).
- After changes: `python onnx/defs/gen_doc.py` and `lintrunner -a --output oneline`. DCO: `git commit -s`.
