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

## PR labels

Before considering an agent-authored PR complete, add the existing labels that
accurately describe its scope. Every agent-authored PR must have at least one
applicable `topic:` or `module:` label, as enforced by
[`check_pr_label.yml`](workflows/check_pr_label.yml). Do not invent new label
names. For example, use `topic: documentation` for a documentation-only PR and
add any other existing labels that materially apply to the change.

We use pixi tasks for installing and linting

```sh
# Compile and install the project
pixi run install
# Run lints and apply fixes
pixi run lint
```
