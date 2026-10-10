# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

CONTRIBUTIONS_QUERY = """
query(
  $login: String!,
  $org: ID!,
  $from: DateTime!,
  $to: DateTime!,
  $issueAfter: String,
  $prAfter: String,
  $reviewAfter: String,
  $repoAfter: String
) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to, organizationID: $org) {
      restrictedContributionsCount
      totalCommitContributions
      totalIssueContributions
      totalPullRequestContributions
      totalPullRequestReviewContributions
      commitContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
        contributions(first: 1) {
          totalCount
          nodes { occurredAt }
        }
      }
      issueContributions(first: 100, after: $issueAfter) {
        pageInfo { hasNextPage endCursor }
        nodes {
          occurredAt
          issue { repository { nameWithOwner } }
        }
      }
      pullRequestContributions(first: 100, after: $prAfter) {
        pageInfo { hasNextPage endCursor }
        nodes {
          occurredAt
          pullRequest { repository { nameWithOwner } }
        }
      }
      pullRequestReviewContributions(first: 100, after: $reviewAfter) {
        pageInfo { hasNextPage endCursor }
        nodes {
          occurredAt
          pullRequest { repository { nameWithOwner } }
        }
      }
      repositoryContributions(first: 100, after: $repoAfter) {
        pageInfo { hasNextPage endCursor }
        nodes {
          occurredAt
          repository { nameWithOwner }
        }
      }
    }
  }
}
"""

ORG_QUERY = """
query($org: String!, $after: String) {
  organization(login: $org) {
    id
    membersWithRole(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      edges {
        role
        node { login }
      }
    }
  }
}
"""

TEAMS_QUERY = """
query($org: String!, $after: String) {
  organization(login: $org) {
    teams(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        slug
        members(first: 100) {
          pageInfo { hasNextPage }
          nodes { login }
        }
      }
    }
  }
}
"""

LAST_ACTIVITY_QUERY = """
query($login: String!, $org: ID!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to, organizationID: $org) {
      commitContributionsByRepository(maxRepositories: 100) {
        contributions(first: 1) { nodes { occurredAt } }
      }
      issueContributions(first: 1) { nodes { occurredAt } }
      pullRequestContributions(first: 1) { nodes { occurredAt } }
      pullRequestReviewContributions(first: 1) { nodes { occurredAt } }
      repositoryContributions(first: 1) { nodes { occurredAt } }
    }
  }
}
"""

ACTIVITY_FIELDS = (
    "commits",
    "issues",
    "pull_requests",
    "reviews",
    "repositories_created",
)
RECENT_ACTIVITY_DAYS = 180
REVIEW_ACTIVITY_DAYS = 365


def run_gh(arguments: list[str]) -> Any:
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError("GitHub CLI executable 'gh' was not found")
    completed = subprocess.run(  # noqa: S603
        [gh, *arguments],
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


def iso_datetime(value: date, *, end_of_day: bool = False) -> str:
    time = "23:59:59Z" if end_of_day else "00:00:00Z"
    return f"{value.isoformat()}T{time}"


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


def split_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=365), end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def list_members(org: str) -> tuple[str, list[dict[str, str]]]:
    members: list[dict[str, str]] = []
    cursor = None
    org_id = ""
    while True:
        data = graphql(ORG_QUERY, {"org": org, "after": cursor})
        organization = data.get("organization")
        if organization is None:
            raise RuntimeError(f"Organization {org!r} was not found or is not visible")
        org_id = organization["id"]
        connection = organization["membersWithRole"]
        members.extend(
            {"login": edge["node"]["login"], "org_role": edge["role"].lower()}
            for edge in connection["edges"]
        )
        if not connection["pageInfo"]["hasNextPage"]:
            break
        cursor = connection["pageInfo"]["endCursor"]
    return org_id, sorted(members, key=lambda member: member["login"].lower())


def list_repositories(org: str) -> list[dict[str, Any]]:
    repositories = rest_pages(
        f"/orgs/{org}/repos?type=all&sort=full_name&direction=asc&per_page=100"
    )
    return [
        {
            "name_with_owner": repository["full_name"],
            "archived": repository["archived"],
            "fork": repository["fork"],
            "visibility": repository["visibility"],
        }
        for repository in repositories
    ]


def list_team_memberships(org: str) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    cursor = None
    while True:
        data = graphql(TEAMS_QUERY, {"org": org, "after": cursor})
        organization = data.get("organization")
        if organization is None:
            raise RuntimeError(f"Organization {org!r} was not found or is not visible")
        teams = organization["teams"]
        for team in teams["nodes"]:
            if team["members"]["pageInfo"]["hasNextPage"]:
                raise RuntimeError(
                    f"Team {team['slug']!r} has more than 100 members; "
                    "nested team pagination is required"
                )
            for member in team["members"]["nodes"]:
                memberships.setdefault(member["login"], []).append(team["slug"])
        if not teams["pageInfo"]["hasNextPage"]:
            break
        cursor = teams["pageInfo"]["endCursor"]
    return {login: sorted(slugs) for login, slugs in memberships.items()}


def empty_activity() -> dict[str, Any]:
    return {
        "commits": 0,
        "issues": 0,
        "pull_requests": 0,
        "reviews": 0,
        "repositories_created": 0,
        "last_activity_at": None,
    }


def update_last_activity(activity: dict[str, Any], occurred_at: str | None) -> None:
    if occurred_at and (
        activity["last_activity_at"] is None
        or occurred_at > activity["last_activity_at"]
    ):
        activity["last_activity_at"] = occurred_at


def add_connection_nodes(
    activity_by_repo: dict[str, dict[str, Any]],
    connection: dict[str, Any],
    field: str,
    subject: str,
) -> None:
    for node in connection["nodes"]:
        repository = node[subject]["repository"]["nameWithOwner"]
        activity = activity_by_repo.setdefault(repository, empty_activity())
        activity[field] += 1
        update_last_activity(activity, node["occurredAt"])


def fetch_window_activity(
    login: str,
    org_id: str,
    start: date,
    end: date,
) -> tuple[dict[str, dict[str, Any]], int]:
    activity_by_repo: dict[str, dict[str, Any]] = {}
    cursors: dict[str, str | None] = {
        "issueAfter": None,
        "prAfter": None,
        "reviewAfter": None,
        "repoAfter": None,
    }
    restricted = 0
    first_page = True
    connections = {
        "issueAfter": ("issueContributions", "issues", "issue"),
        "prAfter": ("pullRequestContributions", "pull_requests", "pullRequest"),
        "reviewAfter": (
            "pullRequestReviewContributions",
            "reviews",
            "pullRequest",
        ),
        "repoAfter": (
            "repositoryContributions",
            "repositories_created",
            "repository",
        ),
    }
    pending_connections = set(connections)

    while True:
        data = graphql(
            CONTRIBUTIONS_QUERY,
            {
                "login": login,
                "org": org_id,
                "from": iso_datetime(start),
                "to": iso_datetime(end, end_of_day=True),
                **cursors,
            },
        )
        user = data.get("user")
        if user is None:
            raise RuntimeError(
                f"Organization member {login!r} no longer has a GitHub account"
            )
        collection = user["contributionsCollection"]

        if first_page:
            restricted = collection["restrictedContributionsCount"]
            for contribution in collection["commitContributionsByRepository"]:
                repository = contribution["repository"]["nameWithOwner"]
                commits = contribution["contributions"]
                activity = activity_by_repo.setdefault(repository, empty_activity())
                activity["commits"] += commits["totalCount"]
                if commits["nodes"]:
                    update_last_activity(activity, commits["nodes"][0]["occurredAt"])
            first_page = False

        for cursor_name, (connection_name, field, subject) in connections.items():
            if cursor_name not in pending_connections:
                continue
            connection = collection[connection_name]
            if subject == "repository":
                for node in connection["nodes"]:
                    repository = node["repository"]["nameWithOwner"]
                    activity = activity_by_repo.setdefault(repository, empty_activity())
                    activity[field] += 1
                    update_last_activity(activity, node["occurredAt"])
            else:
                add_connection_nodes(activity_by_repo, connection, field, subject)
            page_info = connection["pageInfo"]
            if page_info["hasNextPage"]:
                cursors[cursor_name] = page_info["endCursor"]
            else:
                pending_connections.remove(cursor_name)
        if not pending_connections:
            break

    return activity_by_repo, restricted


def merge_activity(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in ACTIVITY_FIELDS:
        target[field] += source[field]
    update_last_activity(target, source["last_activity_at"])


def fetch_last_activity(
    login: str,
    org_id: str,
    windows: list[tuple[date, date]],
) -> str | None:
    for start, end in reversed(windows):
        data = graphql(
            LAST_ACTIVITY_QUERY,
            {
                "login": login,
                "org": org_id,
                "from": iso_datetime(start),
                "to": iso_datetime(end, end_of_day=True),
            },
        )
        user = data.get("user")
        if user is None:
            raise RuntimeError(
                f"Organization member {login!r} no longer has a GitHub account"
            )
        collection = user["contributionsCollection"]
        occurred_at = [
            node["occurredAt"]
            for connection_name in (
                "issueContributions",
                "pullRequestContributions",
                "pullRequestReviewContributions",
                "repositoryContributions",
            )
            for node in collection[connection_name]["nodes"]
        ]
        occurred_at.extend(
            node["occurredAt"]
            for contribution in collection["commitContributionsByRepository"]
            for node in contribution["contributions"]["nodes"]
        )
        if occurred_at:
            return max(occurred_at)
    return None


def fetch_member_activity(
    org_id: str,
    member: dict[str, str],
    teams: list[str],
    windows: list[tuple[date, date]],
    historical_windows: list[tuple[date, date]],
) -> dict[str, Any]:
    activity_by_repo: dict[str, dict[str, Any]] = {}
    restricted = 0
    for start, end in windows:
        window_activity, window_restricted = fetch_window_activity(
            member["login"], org_id, start, end
        )
        restricted += window_restricted
        for repository, activity in window_activity.items():
            merge_activity(
                activity_by_repo.setdefault(repository, empty_activity()), activity
            )
    scoring_period_last_activity_at = max(
        (
            activity["last_activity_at"]
            for activity in activity_by_repo.values()
            if activity["last_activity_at"] is not None
        ),
        default=None,
    )
    last_activity_at = scoring_period_last_activity_at or fetch_last_activity(
        member["login"], org_id, historical_windows
    )
    return {
        **member,
        "teams": teams,
        "restricted_contributions": restricted,
        "repositories": activity_by_repo,
        "last_activity_at": last_activity_at,
    }


def activity_score(activity: dict[str, Any]) -> int:
    return (
        activity["commits"]
        + 2 * activity["issues"]
        + 3 * activity["pull_requests"]
        + 2 * activity["reviews"]
        + 3 * activity["repositories_created"]
    )


def review_band(last_activity_at: str | None, report_end: date) -> str:
    if last_activity_at is None:
        return "priority-review"
    last_activity = datetime.fromisoformat(
        last_activity_at.replace("Z", "+00:00")
    ).date()
    inactive_days = (report_end - last_activity).days
    if inactive_days <= RECENT_ACTIVITY_DAYS:
        return "recent"
    if inactive_days <= REVIEW_ACTIVITY_DAYS:
        return "review"
    return "priority-review"


def summarize_member(member: dict[str, Any], report_end: date) -> dict[str, Any]:
    total = empty_activity()
    for activity in member["repositories"].values():
        merge_activity(total, activity)
    return {
        "login": member["login"],
        "org_role": member["org_role"],
        "teams": ";".join(member["teams"]),
        **{field: total[field] for field in ACTIVITY_FIELDS},
        "restricted_contributions": member["restricted_contributions"],
        "activity_score": activity_score(total),
        "scoring_period_last_activity_at": total["last_activity_at"],
        "last_activity_at": member["last_activity_at"],
        "review_band": review_band(member["last_activity_at"], report_end),
    }


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
    repositories: list[dict[str, Any]],
    members: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_member(member, end) for member in members]
    summaries.sort(
        key=lambda row: (
            row["last_activity_at"] is not None,
            row["last_activity_at"] or "",
            row["login"].lower(),
        )
    )

    repository_rows = []
    for member in members:
        known_repositories = member["repositories"]
        for repository in repositories:
            activity = known_repositories.get(
                repository["name_with_owner"], empty_activity()
            )
            repository_rows.append(
                {
                    "login": member["login"],
                    "repository": repository["name_with_owner"],
                    "visibility": repository["visibility"],
                    "archived": repository["archived"],
                    "fork": repository["fork"],
                    **{field: activity[field] for field in ACTIVITY_FIELDS},
                    "activity_score": activity_score(activity),
                    "last_activity_at": activity["last_activity_at"],
                }
            )

    summary_fields = [
        "login",
        "org_role",
        "teams",
        *ACTIVITY_FIELDS,
        "restricted_contributions",
        "activity_score",
        "scoring_period_last_activity_at",
        "last_activity_at",
        "review_band",
    ]
    repository_fields = [
        "login",
        "repository",
        "visibility",
        "archived",
        "fork",
        *ACTIVITY_FIELDS,
        "activity_score",
        "last_activity_at",
    ]
    write_csv(output_dir / "member-summary.csv", summaries, summary_fields)
    write_csv(
        output_dir / "member-repository-activity.csv",
        repository_rows,
        repository_fields,
    )

    report = {
        "metadata": {
            "organization": org,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            "member_count": len(members),
            "repository_count": len(repositories),
            "score": (
                "commits + 2*issues + 3*pull_requests + 2*reviews "
                "+ 3*repositories_created"
            ),
        },
        "members": summaries,
        "member_repository_activity": repository_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    priority = [row for row in summaries if row["review_band"] == "priority-review"]
    review = [row for row in summaries if row["review_band"] == "review"]
    lines = [
        f"# {org} organization access review",
        "",
        f"Period: {start.isoformat()} through {end.isoformat()}",
        "",
        f"Members: {len(members)} | Repositories: {len(repositories)}",
        "",
        "This report measures observable activity inside the organization. It is a manual",
        "access-review aid and does not recommend automatic removal.",
        "",
        "## Review queue",
        "",
        "| Account | Role | Teams | Score | Last observable activity | Band |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in [*priority, *review]:
        team_count = len(row["teams"].split(";")) if row["teams"] else 0
        lines.append(
            f"| {row['login']} | {row['org_role']} | {team_count} | "
            f"{row['activity_score']} | {row['last_activity_at'] or 'none'} | "
            f"{row['review_band']} |"
        )
    if not priority and not review:
        lines.append("| _No accounts flagged_ | | | | | |")
    lines.extend(
        [
            "",
            "## All members by last observable activity",
            "",
            "Accounts with no observable activity appear first, followed by oldest-to-newest",
            "activity.",
            "",
            "| Account | Role | Teams | Score | Last observable activity | Band |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in summaries:
        team_count = len(row["teams"].split(";")) if row["teams"] else 0
        lines.append(
            f"| {row['login']} | {row['org_role']} | {team_count} | "
            f"{row['activity_score']} | {row['last_activity_at'] or 'none'} | "
            f"{row['review_band']} |"
        )
    lines.extend(
        [
            "",
            "## Methodology and limitations",
            "",
            (
                "- `recent`: activity within 180 days; `review`: 181-365 days; "
                "`priority-review`: over 365 days or no observed activity."
            ),
            (
                "- The score is `commits + 2*issues + 3*pull requests + 2*reviews "
                "+ 3*repositories created`."
            ),
            "- Activity outside the organization is intentionally excluded because it does not",
            "  demonstrate a need for organization-level access.",
            "- GitHub contributions do not include comments, Discussions, moderation, security",
            "  work, draft work, or every commit outside a repository's default branch.",
            "- Last observable activity is not a GitHub login timestamp.",
            "- Last activity is searched across the preceding 10 years; activity counts and",
            "  scores cover only the stated report period.",
            "- Check governance responsibilities, leave, bot/service accounts, and contact the",
            "  member before changing access.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a GitHub organization access-review report."
    )
    parser.add_argument("--org", required=True, help="GitHub organization login")
    parser.add_argument(
        "--months",
        type=int,
        default=18,
        help="Number of months to review (default: 18)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="Inclusive end date in YYYY-MM-DD form (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for private report files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent member requests (default: 4)",
    )
    arguments = parser.parse_args()
    if arguments.months < 1:
        parser.error("--months must be positive")
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    return arguments


def main() -> int:
    arguments = parse_args()
    start = subtract_months(arguments.end_date, arguments.months)
    windows = split_windows(start, arguments.end_date)
    historical_start = subtract_months(arguments.end_date, 120)
    historical_end = start - timedelta(days=1)
    historical_windows = (
        split_windows(historical_start, historical_end)
        if historical_start <= historical_end
        else []
    )
    org_id, member_stubs = list_members(arguments.org)
    repositories = list_repositories(arguments.org)
    team_memberships = list_team_memberships(arguments.org)

    members = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                fetch_member_activity,
                org_id,
                member,
                team_memberships.get(member["login"], []),
                windows,
                historical_windows,
            ): member["login"]
            for member in member_stubs
        }
        for future in as_completed(futures):
            login = futures[future]
            try:
                members.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"Failed to collect activity for {login!r}"
                ) from error

    members.sort(key=lambda member: member["login"].lower())
    write_report(
        arguments.output_dir,
        arguments.org,
        start,
        arguments.end_date,
        repositories,
        members,
    )
    print(  # noqa: T201
        f"Wrote access review for {len(members)} members and "
        f"{len(repositories)} repositories to {arguments.output_dir}"
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
        print(f"error: {'; caused by: '.join(messages)}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
