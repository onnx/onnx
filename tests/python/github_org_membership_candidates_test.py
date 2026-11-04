# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / ".agents"
    / "skills"
    / "github-org-membership-candidates"
    / "scripts"
    / "github_org_membership_candidates.py"
)
if not SCRIPT_PATH.is_file():
    pytest.skip(
        "Repository-only skill is not included in source distributions",
        allow_module_level=True,
    )

SPEC = importlib.util.spec_from_file_location(
    "github_org_membership_candidates", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
membership_candidates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = membership_candidates
SPEC.loader.exec_module(membership_candidates)


def test_issue_search_splits_windows_above_result_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = []

    def run_gh(arguments: list[str]) -> dict:
        query = parse_qs(urlsplit(arguments[1]).query)["q"][0]
        queries.append(query)
        if query.endswith("2026-01-01..2026-01-04"):
            return {"total_count": 1500, "items": []}
        return {"total_count": 1, "items": [{"query": query}]}

    monkeypatch.setattr(membership_candidates, "run_gh", run_gh)

    items, capped = membership_candidates.search_in_windows(
        membership_candidates.search_issues,
        "org:onnx is:issue",
        "created",
        date(2026, 1, 1),
        date(2026, 1, 4),
    )

    assert not capped
    assert queries == [
        "org:onnx is:issue created:2026-01-01..2026-01-04",
        "org:onnx is:issue created:2026-01-01..2026-01-02",
        "org:onnx is:issue created:2026-01-03..2026-01-04",
    ]
    assert [item["query"] for item in items] == queries[1:]


def test_single_day_above_result_cap_is_reported_as_capped() -> None:
    items, capped = membership_candidates.search_in_windows(
        lambda _query, _splittable: ([{"id": 1}], 1500),
        "org:onnx is:issue",
        "created",
        date(2026, 1, 1),
        date(2026, 1, 1),
    )

    assert capped
    assert items == [{"id": 1}]


def review(login: str, submitted_at: str, account_type: str = "User") -> dict:
    return {
        "submittedAt": submitted_at,
        "author": {"__typename": account_type, "login": login},
    }


def pull_request(
    author: str,
    created_at: str,
    merged_at: str | None,
    reviews: list[dict],
    is_draft: bool = False,
) -> dict:
    return {
        "number": 1,
        "createdAt": created_at,
        "mergedAt": merged_at,
        "isDraft": is_draft,
        "author": {"__typename": "User", "login": author},
        "repository": {"nameWithOwner": "onnx/onnx"},
        "reviews": {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": reviews,
        },
    }


def test_pull_request_activity_counts_merges_and_reviews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulls = [
        pull_request(
            "contributor",
            "2026-02-01T00:00:00Z",
            "2026-02-03T00:00:00Z",
            [
                review("reviewer", "2026-02-02T00:00:00Z"),
                review("reviewer", "2026-02-02T12:00:00Z"),
                review("contributor", "2026-02-02T00:00:00Z"),
                review("member", "2026-02-02T00:00:00Z"),
                review("copilot-pull-request-reviewer", "2026-02-02T00:00:00Z", "Bot"),
            ],
        ),
        # Opened before the period, merged inside it.
        pull_request("contributor", "2025-12-01T00:00:00Z", "2026-01-05T00:00:00Z", []),
        pull_request("contributor", "2026-03-01T00:00:00Z", None, [], is_draft=True),
        pull_request("justinchuby-bot", "2026-03-01T00:00:00Z", None, []),
    ]
    monkeypatch.setattr(
        membership_candidates,
        "search_in_windows",
        lambda *_arguments: (pulls, False),
    )
    candidates = {}

    capped = membership_candidates.collect_pull_request_activity(
        "onnx", date(2026, 1, 1), date(2026, 10, 1), {"member"}, candidates
    )

    assert not capped
    assert set(candidates) == {"contributor", "reviewer"}
    contributor = candidates["contributor"]
    assert contributor["pull_requests"] == 1
    assert contributor["merged_pull_requests"] == 2
    assert contributor["reviewed_pull_requests"] == 0
    reviewer = candidates["reviewer"]
    assert reviewer["reviewed_pull_requests"] == 1
    assert reviewer["last_activity_at"] == "2026-02-02T12:00:00Z"


def test_bot_detection_covers_user_typed_automation() -> None:
    assert membership_candidates.is_bot("actions-user", "User")
    assert membership_candidates.is_bot("justinchuby-bot", "User")
    assert membership_candidates.is_bot("dependabot[bot]", "Bot")
    assert not membership_candidates.is_bot("robertknight", "User")


def test_advisory_forks_are_detected() -> None:
    assert membership_candidates.is_advisory_fork("onnx/onnx-ghsa-vc7w-fv4q-mj9g")
    assert not membership_candidates.is_advisory_fork("onnx/onnx-mlir")


def test_commit_in_several_repositories_is_counted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        membership_candidates,
        "fetch_repository_commits",
        lambda _name_with_owner, _since, _until: [
            {
                "sha": "abc123",
                "author": {"login": "contributor", "type": "User"},
                "commit": {"committer": {"date": "2026-10-01T00:00:00Z"}},
            }
        ],
    )
    candidates = {}

    membership_candidates.collect_commit_activity(
        [{"name_with_owner": "onnx/onnx"}, {"name_with_owner": "onnx/onnx-mirror"}],
        date(2026, 9, 1),
        date(2026, 10, 1),
        set(),
        candidates,
        workers=2,
    )

    assert candidates["contributor"]["commits"] == 1
    assert candidates["contributor"]["repo_activity"] == {"onnx/onnx": 1}


def test_list_member_logins_requires_organization_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        membership_candidates,
        "graphql",
        lambda _query, _variables: {
            "organization": {
                "viewerIsAMember": False,
                "membersWithRole": {
                    "nodes": [{"login": "public-member"}],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                },
            }
        },
    )

    with pytest.raises(RuntimeError, match="must be a member"):
        membership_candidates.list_member_logins("onnx")


def test_collect_commit_activity_uses_committer_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        membership_candidates,
        "fetch_repository_commits",
        lambda _name_with_owner, _since, _until: [
            {
                "author": {"login": "contributor", "type": "User"},
                "commit": {
                    "author": {"date": "2020-01-01T00:00:00Z"},
                    "committer": {"date": "2026-10-01T00:00:00Z"},
                },
            }
        ],
    )
    candidates = {}

    membership_candidates.collect_commit_activity(
        [{"name_with_owner": "onnx/onnx"}],
        date(2026, 9, 1),
        date(2026, 10, 1),
        set(),
        candidates,
        workers=1,
    )

    assert candidates["contributor"]["first_activity_at"] == "2026-10-01T00:00:00Z"


def test_primary_repo_tie_is_deterministic() -> None:
    candidate = membership_candidates.empty_candidate("contributor")
    candidate["repo_activity"] = {"onnx/z-repo": 2, "onnx/a-repo": 2}

    assert membership_candidates.primary_repo(candidate) == "onnx/a-repo"


def test_short_report_window_is_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "github_org_membership_candidates.py",
            "--org",
            "onnx",
            "--months",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    arguments = membership_candidates.parse_args()

    assert arguments.months == 1
    assert arguments.min_active_months > arguments.months
