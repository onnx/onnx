---
name: github-org-membership-candidates
description: Find GitHub-active non-members of an organization who may be candidates for organization membership. Use when identifying contributors to nominate, sponsor, or invite, or when preparing input for a Contributor nomination.
---

Use `scripts/github_org_membership_candidates.py` to find people who contribute to a GitHub
organization's repositories but are not organization members, ranked by observable activity. The
script reads data with the authenticated `gh` CLI and writes CSV, JSON, and Markdown reports.

This is the mirror of the
[github-org-access-review](../github-org-access-review/SKILL.md) skill: that skill looks for
inactive *members* who may no longer need access; this skill looks for active *non-members* who
may be worth inviting.

## Principles

- Measure activity only inside the selected GitHub organization.
- Never automatically nominate, sponsor, or add anyone. Flag accounts for manual review by
  people who can judge sponsorship, company affiliation, and standing.
- Keep generated reports outside the repository because contributor activity data may be
  sensitive.
- A high score is evidence worth reviewing, not a decision.

## Prerequisites

The authenticated account must be a member of the organization so the report can exclude members
whose organization membership is private. Organization ownership is not required. Authenticate
with at least `read:org` and `repo` (for private repository activity).

Verify the active account without querying any individual member:

```bash
gh auth status
```

## Generate a report

Use an absolute output directory outside the repository:

```bash
python .agents/skills/github-org-membership-candidates/scripts/github_org_membership_candidates.py \
  --org onnx \
  --months 12 \
  --output-dir <private-output-directory>
```

The generated files are:

| File | Content |
|---|---|
| `candidate-summary.csv` | Non-member login, contribution counts (issues, opened / merged / reviewed pull requests, commits), score, activity span, a best-effort primary repository / likely SIG-WG guess, and candidate band |
| `report.json` | Machine-readable report and metadata |
| `report.md` | Human-readable candidate queue and methodology |

## Interpretation

The default score is:

```text
commits + issues + 3 * merged_pull_requests + 2 * reviewed_pull_requests
```

Opened pull requests are reported but not scored, so unmerged or rejected pull requests add
nothing; reviewing other people's pull requests, which is usually the clearest sign of sustained
engagement, is. `reviewed_pull_requests` counts distinct pull requests by someone else with at
least one review (comment-only reviews included) submitted by the account in the period. Issues
weigh the same as a commit so that batches of tool-generated bug reports do not outrank merged
work.

The score is a sorting aid, not a judgment of contribution quality or eligibility. Candidate
bands:

- `priority-candidate`: score at or above `--min-score` (default 8) **and** an activity span of at
  least `--min-active-months` (default 6, converted to calendar days relative to the report's end
  date) between the candidate's first and last observed contribution in the reporting period
- `candidate`: any observable activity that does not clear both bars

The span requirement exists because a raw contribution count, or even "touches two different
calendar quarters," rewards a short burst (e.g. a person active only in late March and early
April crosses a quarter boundary after about a day). Requiring the first and last observed
contribution to be several months apart is a more reliable signal that someone has been
contributing steadily rather than in a single sprint.

The default `--months 12` mirrors the "active in the last 12 months" removal bar for Contributors
in `community/readme.md`'s Community Roles section (long-standing and already ratified, not part
of any open PR). That same section requires sponsorship by two Approvers from different companies
to become a governance *Contributor* (voting rights, issue/PR assignment) — but a plain GitHub
organization seat and the *Contributor* role are not necessarily the same thing today: the 1:1
mapping between them, and a formal `Contributor nomination: <handle>` issue process, only exist in
the still-open, explicitly-WIP proposal at
[onnx/onnx#8222](https://github.com/onnx/onnx/pull/8222). In current practice a SIG chair may add
someone to the org for practical repository access without the formal Contributor sponsorship
ceremony. This script does not check sponsorship, company affiliation, or which of these paths
applies, since none of that is reliably derivable from the GitHub API. Treat its output as a
shortlist to bring to prospective sponsors or SIG chairs, not a nomination itself.

## Coverage limitations

- Pull requests, including their reviews, are read through a GraphQL search over pull requests
  *updated* in the period, so a pull request opened earlier but merged or reviewed in the period
  is still seen.
- GitHub's search API caps results at 1,000 per query. The script halves each search's date
  window until every part fits under the cap; only a single day that still exceeds it can be
  undercounted, and the report notes when that happens.
- Commit counts only cover each repository's default branch, and only non-fork, non-archived
  repositories. GitHub Security Advisory temporary private forks (`<repo>-ghsa-...`) are skipped:
  GitHub does not flag them as forks, but they carry a full copy of the upstream history. A
  commit that still appears in several scanned repositories is counted once.
- Review comments count through their review; other comments, Discussions, draft work,
  moderation, security work, and activity outside GitHub are not counted.
- Bot accounts are excluded on a best-effort basis: GitHub `type: Bot`, the `actions-user`
  identity used by GitHub Actions, and logins ending in `-bot` or `[bot]`.
- The `Likely SIG/WG` guess is repository-level only, from a mapping hand-built from
  [onnx/sigs](https://github.com/onnx/sigs)'s responsibilities table (no machine-readable
  repository-to-SIG registry exists). For `onnx/onnx` specifically, Architecture & Infra is the
  defacto owner but Operators owns the operator schema/spec portion of that same repository —
  the report flags this ambiguity rather than guessing further. Confirm with SIG chairs before
  assigning a Team.
