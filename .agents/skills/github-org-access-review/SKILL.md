---
name: github-org-access-review
description: Generate a GitHub organization access-review report from organization-internal activity. Use when auditing members, repository activity, roles, teams, inactivity, or candidates for manual access review.
---

Use `scripts/github_org_access_review.py` to generate an evidence-based access review for a
GitHub organization. The script reads data with the authenticated `gh` CLI and writes CSV,
JSON, and Markdown reports.

## Principles

- Measure activity only inside the selected GitHub organization.
- Treat external work as valuable but not as evidence that organization-level access is needed.
- Never make an automatic removal decision. Flag accounts for manual review.
- Keep generated reports outside the repository because membership, role, and team data may be
  sensitive.
- Describe "last active" as the last observable GitHub contribution, not the last login time.

## Prerequisites

The active `gh` account must be an organization owner and its token must have access to all
repositories being reviewed. At minimum, authenticate with `read:org`; private repository
activity also requires `repo`.

Verify the active account without querying any individual member:

```bash
gh auth status
```

## Generate a report

Use an absolute output directory outside the repository:

```bash
python .agents/skills/github-org-access-review/scripts/github_org_access_review.py \
  --org onnx \
  --months 18 \
  --output-dir <private-output-directory>
```

The generated files are:

| File | Content |
|---|---|
| `member-summary.csv` | Member role, teams, contribution totals, score, historical last activity, and review band |
| `member-repository-activity.csv` | One row per member and organization repository |
| `report.json` | Machine-readable report and metadata |
| `report.md` | Human-readable review queue and methodology |

## Interpretation

The default score is:

```text
commits + 3 * pull_requests + 2 * reviews + 2 * issues + 3 * repositories_created
```

The score is a sorting aid, not a judgment of contribution quality. Review bands use recency:

- `recent`: activity within 180 days
- `review`: last activity 181-365 days ago
- `priority-review`: no observable activity or last activity more than 365 days ago

Counts and scores cover the selected reporting period. Last activity is searched separately
across the preceding 10 years so an inactive member's historical date is not incorrectly shown
as `none`.

Before changing access, review team responsibilities, ownership, security and release duties,
temporary leave, and whether the account is a bot or service account. Follow the organization's
documented governance and contact the member before removal.

## Coverage limitations

GitHub's contribution data includes commits to default branches, opened issues and pull
requests, pull-request reviews, and repository creation. It does not represent every useful
action: issue or pull-request comments, Discussions, draft work, moderation, security work, and
activity outside GitHub are not counted. Private contributions depend on token visibility.
