## PR descriptions

As a starting point, agent-authored PRs should structure their description with the following, filling in whatever applies (this isn't yet a project-wide requirement for human contributors):

- **Root cause** (bug fixes) / **What changed** (features): what was actually wrong, or what this adds.
- **Fix approach**: what was done, and any alternatives considered.
- **Spec/compatibility impact**: does this change operator schemas, opset versions, the IR spec, or a public API in a way that could be backward-incompatible?
- **Security impact**: does this touch model loading, external data handling, or parsing of untrusted input? See [SECURITY.md](../SECURITY.md).
- **Tests**: what was added, modified, or removed.
- **Auto-generated files**: were `docs/Operators.md`, `docs/Changelog.md`, protobuf files, or test data regenerated where applicable? See CLAUDE.md's Auto-Generated Files table.
- **Backport**: does this need to land in an already-released version? If so, note the minimum affected version — see [RELEASE-MANAGEMENT.md](../RELEASE-MANAGEMENT.md)'s Long-Term Support section for the case-by-case backport policy.

We use lintrunner as the linter:

```sh
# Display all lints and apply the fixes
lintrunner -a --output oneline
# Or apply fixes only (faster)
lintrunner f --output oneline
```

To build ONNX:

```sh
python -m pip install --quiet --upgrade pip setuptools wheel
export ONNX_BUILD_TESTS=0
export ONNX_ML=1
python -m pip install .
```
