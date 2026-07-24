## PR titles

PRs are squash-merged, so the **PR title becomes the commit message on `main`**. As a starting point, agent-authored PRs should use [Conventional Commits](https://www.conventionalcommits.org/) for their title — this isn't yet a project-wide requirement for human contributors:

```
<type>(<scope>): <description>
```

- `type`: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, or `revert`
- `scope`: optional, e.g. the affected area (`defs`, `shape_inference`, `ci`, `docs`)
- `description`: imperative mood, lower case, no trailing period

Example: `fix(shape_inference): handle negative axis in Squeeze`

When a change could fit more than one type, prefer `ci` for anything under `.github/workflows`, `build` for build-system/toolchain/dependency-version changes, and `chore` for everything else non-user-facing (dead code removal, repo maintenance).

Individual commits within the PR do not need to follow this format, but each must still carry a DCO sign-off.

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
