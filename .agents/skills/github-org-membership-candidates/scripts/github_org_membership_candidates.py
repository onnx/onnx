# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

ORG_MEMBERS_QUERY = """
query($org: String!, $after: String) {
  organization(login: $org) {
    membersWithRole(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { login }
    }
  }
}
"""

SEARCH_PAGE_SIZE = 100
SEARCH_MAX_PAGES = 10  # GitHub search API caps results at 1,000 per query.


def run_gh(arguments: list[str]) -> Any:
    completed = subprocess.run(  # noqa: S603
        ["gh", *arguments],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"gh {' '.join(arguments[:2])} failed: {message}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("GitHub CLI returned invalid JSON") from error


def graphql(query: str, variables: dict[str, str | None]) -> dict[str, Any]:
    arguments = ["api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is not None:
            arguments.extend(["-F", f"{key}={value}"])
    result = run_gh(arguments)
    if result.get("errors"):
        raise RuntimeError(f"GitHub GraphQL error: {json.dumps(result['errors'])}")
    return result["data"]


def rest_pages(path: str) -> list[dict[str, Any]]:
    pages = run_gh(["api", "--paginate", "--slurp", path])
    return [item for page in pages for item in page]


def list_member_logins(org: str) -> set[str]:
    members: set[str] = set()
    cursor = None
    while True:
        data = graphql(ORG_MEMBERS_QUERY, {"org": org, "after": cursor})
        organization = data.get("organization")
        if organization is None:
            raise RuntimeError(f"Organization {org!r} was not found or is not visible")
        connection = organization["membersWithRole"]
        members.update(node["login"] for node in connection["nodes"])
        if not connection["pageInfo"]["hasNextPage"]:
            break
        cursor = connection["pageInfo"]["endCursor"]
    return members


def list_repositories(org: str) -> list[dict[str, Any]]:
    repositories = rest_pages(
        f"/orgs/{org}/repos?type=all&sort=full_name&direction=asc&per_page=100"
    )
    return [
        {
            "name_with_owner": repository["full_name"],
            "archived": repository["archived"],
            "fork": repository["fork"],
        }
        for repository in repositories
    ]


def empty_candidate(login: str) -> dict[str, Any]:
    return {
        "login": login,
        "issues": 0,
        "pull_requests": 0,
        "commits": 0,
        "last_activity_at": None,
        "search_capped": False,
    }


def note_activity(
    candidates: dict[str, dict[str, Any]],
    login: str,
    field: str,
    occurred_at: str,
) -> None:
    candidate = candidates.setdefault(login, empty_candidate(login))
    candidate[field] += 1
    if (
        candidate["last_activity_at"] is None
        or occurred_at > candidate["last_activity_at"]
    ):
        candidate["last_activity_at"] = occurred_at


def search_issues_or_pulls(
    org: str,
    item_type: str,
    start: date,
    end: date,
) -> tuple[list[dict[str, Any]], bool]:
    query = f"org:{org} type:{item_type} created:{start.isoformat()}..{end.isoformat()}"
    items: list[dict[str, Any]] = []
    for page in range(1, SEARCH_MAX_PAGES + 1):
        params = {
            "q": query,
            "per_page": SEARCH_PAGE_SIZE,
            "page": page,
            "sort": "created",
            "order": "asc",
        }
        result = run_gh(["api", f"search/issues?{urlencode(params)}"])
        page_items = result.get("items", [])
        items.extend(page_items)
        if len(page_items) < SEARCH_PAGE_SIZE:
            return items, False
    return items, True


def collect_issue_and_pull_activity(
    org: str,
    start: date,
    end: date,
    member_logins: set[str],
) -> tuple[dict[str, dict[str, Any]], bool]:
    candidates: dict[str, dict[str, Any]] = {}
    capped = False
    for item_type, field in (("issue", "issues"), ("pr", "pull_requests")):
        items, hit_cap = search_issues_or_pulls(org, item_type, start, end)
        capped = capped or hit_cap
        for item in items:
            user = item.get("user") or {}
            login = user.get("login")
            if not login or login in member_logins or user.get("type") == "Bot":
                continue
            note_activity(candidates, login, field, item["created_at"])
    return candidates, capped


def collect_commit_activity(
    repositories: list[dict[str, Any]],
    start: date,
    end: date,
    member_logins: set[str],
    candidates: dict[str, dict[str, Any]],
) -> None:
    since = f"{start.isoformat()}T00:00:00Z"
    until = f"{end.isoformat()}T23:59:59Z"
    for repository in repositories:
        if repository["fork"] or repository["archived"]:
            continue
        params = {"since": since, "until": until, "per_page": 100}
        try:
            commits = rest_pages(
                f"/repos/{repository['name_with_owner']}/commits?{urlencode(params)}"
            )
        except RuntimeError:
            # Empty repositories and repositories without a default branch return an error.
            continue
        for commit in commits:
            author = commit.get("author")
            if not author:
                continue
            login = author.get("login")
            if not login or login in member_logins or author.get("type") == "Bot":
                continue
            note_activity(
                candidates, login, "commits", commit["commit"]["author"]["date"]
            )


def candidate_score(candidate: dict[str, Any]) -> int:
    return (
        candidate["commits"] + 2 * candidate["issues"] + 3 * candidate["pull_requests"]
    )


def candidate_band(score: int, min_score: int) -> str:
    return "priority-candidate" if score >= min_score else "candidate"


def summarize(
    candidates: dict[str, dict[str, Any]], min_score: int
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates.values():
        score = candidate_score(candidate)
        rows.append(
            {
                "login": candidate["login"],
                "issues": candidate["issues"],
                "pull_requests": candidate["pull_requests"],
                "commits": candidate["commits"],
                "score": score,
                "last_activity_at": candidate["last_activity_at"],
                "band": candidate_band(score, min_score),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["score"],
            row["last_activity_at"] or "",
            row["login"].lower(),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_dir: Path,
    org: str,
    start: date,
    end: date,
    min_score: int,
    repository_count: int,
    member_count: int,
    rows: list[dict[str, Any]],
    search_capped: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "login",
        "issues",
        "pull_requests",
        "commits",
        "score",
        "last_activity_at",
        "band",
    ]
    write_csv(output_dir / "candidate-summary.csv", rows, summary_fields)

    report = {
        "metadata": {
            "organization": org,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            "member_count": member_count,
            "repository_count": repository_count,
            "min_score": min_score,
            "score": "commits + 2*issues + 3*pull_requests",
            "search_capped": search_capped,
        },
        "candidates": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    priority = [row for row in rows if row["band"] == "priority-candidate"]
    lines = [
        f"# {org} organization membership candidates",
        "",
        f"Period: {start.isoformat()} through {end.isoformat()}",
        "",
        f"Organization members: {member_count} | Repositories scanned: {repository_count}",
        "",
        "This report lists non-members with observable activity inside the organization. It is",
        "a discovery aid for prospective sponsors and does not nominate or add anyone.",
        "",
    ]
    if search_capped:
        lines.extend(
            [
                "**Note:** at least one issue/pull-request search hit GitHub's 1,000-result cap for",
                "this period. Counts may be undercounted; narrow `--months` and re-run for a",
                "precise picture.",
                "",
            ]
        )
    lines.extend(
        [
            "## Priority candidates",
            "",
            "| Account | Score | Issues | Pull requests | Commits | Last observable activity |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {row['login']} | {row['score']} | {row['issues']} | {row['pull_requests']} | "
        f"{row['commits']} | {row['last_activity_at'] or 'none'} |"
        for row in priority
    )
    if not priority:
        lines.append("| _No accounts above the threshold_ | | | | | |")
    lines.extend(
        [
            "",
            "## All candidates",
            "",
            "| Account | Score | Issues | Pull requests | Commits | Last observable activity | Band |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    lines.extend(
        f"| {row['login']} | {row['score']} | {row['issues']} | {row['pull_requests']} | "
        f"{row['commits']} | {row['last_activity_at'] or 'none'} | {row['band']} |"
        for row in rows
    )
    if not rows:
        lines.append("| _No non-member activity observed_ | | | | | | |")
    lines.extend(
        [
            "",
            "## Methodology and limitations",
            "",
            f"- `priority-candidate`: score at or above {min_score}; `candidate`: any observable",
            "  activity below that threshold.",
            "- The score is `commits + 2*issues + 3*pull requests`.",
            "- Pull request reviews are not counted (no organization-wide search for review",
            "  activity without already knowing the account).",
            "- Commits only cover each repository's default branch, and only non-fork,",
            "  non-archived repositories.",
            "- Comments, Discussions, draft work, moderation, and activity outside GitHub are",
            "  not counted.",
            "- Sponsorship, company affiliation, and Code of Conduct standing are not derivable",
            "  from the GitHub API and must be checked manually before any nomination.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find active non-members of a GitHub organization as membership candidates."
    )
    parser.add_argument("--org", required=True, help="GitHub organization login")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months to look back (default: 12)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="Inclusive end date in YYYY-MM-DD form (default: today)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=8,
        help="Score threshold for the priority-candidate band (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for private report files",
    )
    arguments = parser.parse_args()
    if arguments.months < 1:
        parser.error("--months must be positive")
    return arguments


def subtract_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    month_lengths = (
        31,
        29 if year % 4 == 0 else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    return date(year, month, min(value.day, month_lengths[month - 1]))


def main() -> int:
    arguments = parse_args()
    start = subtract_months(arguments.end_date, arguments.months)
    end = arguments.end_date

    member_logins = list_member_logins(arguments.org)
    repositories = list_repositories(arguments.org)

    candidates, search_capped = collect_issue_and_pull_activity(
        arguments.org, start, end, member_logins
    )
    collect_commit_activity(repositories, start, end, member_logins, candidates)

    rows = summarize(candidates, arguments.min_score)
    write_report(
        arguments.output_dir,
        arguments.org,
        start,
        end,
        arguments.min_score,
        len(repositories),
        len(member_logins),
        rows,
        search_capped,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        messages = []
        current: BaseException | None = error
        while current is not None:
            messages.append(str(current))
            current = current.__cause__
        sys.exit(1)
