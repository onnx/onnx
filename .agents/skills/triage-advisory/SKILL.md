---
name: triage-advisory
description: Triage private ONNX security advisories for discoverability against the disclosure policy. Use when asked whether reported vulnerabilities are "publicly known" (easily found with public tooling) and should therefore be public issues rather than private advisories, or to batch-disposition draft/triage GHSA reports.
---

See also: [SECURITY.md](../../../SECURITY.md)

## Purpose

ONNX's disclosure policy turns on whether a report is easily discoverable. This skill
answers that reproducibly, for one advisory or a batch.

**Is it "publicly known"?** SECURITY.md: *"Bugs, even safety-critical ones, that are
easily discovered using widely available tooling are considered publicly known"* —
those belong in **public issues/PRs**, not private advisories.

## Fetch the advisories

Requires a token with `repo` scope on `onnx/onnx` (grants draft + triage visibility).

```bash
# One line per advisory, grouped by state (triage / draft / published / closed)
gh api /repos/onnx/onnx/security-advisories --paginate \
  --jq '.[] | [.state, .ghsa_id, .severity, (.published_at // "—"), .summary] | @tsv' | sort

# Full write-up (description + PoC) for one advisory
gh api /repos/onnx/onnx/security-advisories/GHSA-xxxx-xxxx-xxxx \
  --jq '"\(.ghsa_id) [\(.severity)]\n\(.summary)\n\n\(.description)"'

# N most recently *created* triage reports
gh api /repos/onnx/onnx/security-advisories --paginate \
  --jq '.[] | select(.state=="triage") | [.created_at, .ghsa_id, .summary] | @tsv' | sort -r | head -5
```

## Discoverability test (blind agent)

The bar: **can a reasonably skilled person find this in a few minutes given a high-level
pointer?** The report earns "private" status only if the *specific* instance is non-obvious.

Spawn one `Explore` agent per advisory (in parallel). Give it a **high-level description**
— subsystem + bug class + rough area — but **withhold** the exact line, the trigger
construction, and the PoC. Those are what it must rediscover. Prompt template:

```
You are doing a quick security-triage exercise on the ONNX codebase at <repo>.
Use ONLY publicly available techniques: reading source, grep, reasoning.
Do NOT look at security advisories, GHSA files, or the reports/ directory.

Task: Someone reported <one-sentence high-level description: subsystem, bug class,
rough mechanism — e.g. "an OOB read in Einsum shape inference where the output-shape
build indexes a dim list with a std::map operator[] lookup on a crafted equation">.

Independently locate the responsible code and confirm the mechanism, as a motivated
researcher who just got this tip. Locating/identifying by reading source is ENOUGH —
no reproducer needed. If you must build/run, use pixi tasks (e.g. `pixi run install`),
never raw pip/cmake.

Report: found? the file:line, the exact mechanism, the trigger, and an HONEST estimate
of difficulty given the tip (trivial / few minutes / hours). Be concise.
```

Verdict: if the agent lands on the right `file:line` + trigger in "trivial"/"few minutes",
the report is **publicly known** → should have been a public issue. Note a strengthening
signal: bugs in subsystems already under continuous public fuzzing (e.g. OSS-Fuzz targets
such as the checker, loader, text parser, shape inference, and version converter) are
discoverable with *no* hint at all.

## Output

Per advisory, produce a one-row verdict: **GHSA · reporter's severity · found? how fast?**
Then a recommended disposition: *downgrade to public issue* (easily discovered) or *retain
private* (specific instance is non-trivial to find). Cite the SECURITY.md clause for each
call.

Optionally check whether the bug is already fixed on the current branch before disposing.
