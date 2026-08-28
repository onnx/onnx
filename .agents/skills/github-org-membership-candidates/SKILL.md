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

Read access to the organization's repositories is sufficient; organization ownership is not
required. Authenticate with at least `read:org` and `repo` (for private repository activity).

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
| `candidate-summary.csv` | Non-member login, contribution counts, score, last activity, and candidate band |
| `report.json` | Machine-readable report and metadata |
| `report.md` | Human-readable candidate queue and methodology |

## Interpretation

The default score is:

```text
commits + 2 * issues + 3 * pull_requests
```

The score is a sorting aid, not a judgment of contribution quality or eligibility. Candidate
bands:

- `priority-candidate`: score at or above `--min-score` (default 8) **and** an activity span of
  at least `--min-active-months` (default 6, measured as a 30-day-month day count) between the
  candidate's first and last observed contribution in the reporting period
- `candidate`: any observable activity that does not clear both bars

The span requirement exists because a raw contribution count, or even "touches two different
calendar quarters," rewards a short burst (e.g. a person active only in late March and early
April crosses a quarter boundary after about a day). Requiring the first and last observed
contribution to be several months apart is a more reliable signal that someone has been
contributing steadily rather than in a single sprint.

The default `--months 12` mirrors the "active in the last 12 months" bar used for the Contributor
rung of the ladder described in the org's (work-in-progress) membership process — see
`community/readme.md` on branch/PR
[onnx/onnx#8222](https://github.com/onnx/onnx/pull/8222). That process currently requires
sponsorship by two Approvers from different companies and a nomination issue titled
`Contributor nomination: <handle>`; this script does not check sponsorship or company
affiliation, since neither is reliably derivable from the GitHub API. Treat its output as a
shortlist to bring to prospective sponsors, not a nomination itself.

## Coverage limitations

- Pull request reviews are not counted in this first version: there is no organization-wide
  search for "reviewed by" activity without already knowing the candidate's login. A reviewer
  who rarely opens issues or pull requests may be undercounted or missed entirely.
- Commit counts only cover each repository's default branch, and only non-fork, non-archived
  repositories.
- GitHub's search API caps results at 1,000 per query. In a very active organization or a long
  `--months` window, issue or pull request counts near that cap may be undercounted; narrow
  `--months` and re-run if a report notes the cap was hit.
- Comments, Discussions, draft work, moderation, security work, and activity outside GitHub are
  not counted.
- Bot accounts (GitHub `type: Bot`) are excluded on a best-effort basis.
