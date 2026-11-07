# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import itertools
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / ".agents"
    / "skills"
    / "github-org-access-review"
    / "scripts"
    / "github_org_access_review.py"
)
if not SCRIPT_PATH.is_file():
    pytest.skip(
        "Repository-only skill is not included in source distributions",
        allow_module_level=True,
    )

SPEC = importlib.util.spec_from_file_location("github_org_access_review", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
access_review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = access_review
SPEC.loader.exec_module(access_review)


def test_split_windows_stay_within_one_year() -> None:
    start, end = date(2025, 4, 30), date(2026, 10, 31)

    windows = access_review.split_windows(start, end)

    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(
        (window_end - window_start).days <= 364 for window_start, window_end in windows
    )
    assert all(
        (next_start - previous_end).days == 1
        for (_, previous_end), (next_start, _) in itertools.pairwise(windows)
    )


def test_graphql_sends_variables_as_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run_gh(arguments: list[str]) -> dict:
        calls.append(arguments)
        return {"data": {}}

    monkeypatch.setattr(access_review, "run_gh", run_gh)

    access_review.graphql("query", {"login": "12345", "after": None})

    assert calls == [["api", "graphql", "-f", "query=query", "-f", "login=12345"]]


def empty_connection() -> dict:
    return {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []}


def test_commit_count_uses_commit_total_not_daily_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verified against the live API: onnx/onnx had 123 daily records holding 220 commits, and
    # contributions.totalCount was 220.
    def graphql(_query: str, _variables: dict) -> dict:
        commits = {
            "repository": {"nameWithOwner": "onnx/onnx"},
            "contributions": {
                "totalCount": 220,
                "nodes": [{"occurredAt": "2026-03-02T00:00:00Z"}],
            },
        }
        return {
            "user": {
                "contributionsCollection": {
                    "restrictedContributionsCount": 0,
                    "commitContributionsByRepository": [commits],
                    "issueContributions": empty_connection(),
                    "pullRequestContributions": empty_connection(),
                    "pullRequestReviewContributions": empty_connection(),
                    "repositoryContributions": empty_connection(),
                }
            }
        }

    monkeypatch.setattr(access_review, "graphql", graphql)

    activity, _ = access_review.fetch_window_activity(
        "member", "org-id", date(2025, 11, 1), date(2026, 10, 31)
    )

    assert activity["onnx/onnx"]["commits"] == 220
    assert activity["onnx/onnx"]["last_activity_at"] == "2026-03-02T00:00:00Z"


def test_review_band_accepts_utc_suffix() -> None:
    assert (
        access_review.review_band("2026-10-01T12:00:00Z", date(2026, 10, 31))
        == "recent"
    )


@pytest.mark.parametrize("output_dir", [Path(), Path(__file__).parents[2] / "reports"])
def test_output_dir_inside_repository_is_rejected(
    monkeypatch: pytest.MonkeyPatch, output_dir: Path
) -> None:
    monkeypatch.chdir(Path(__file__).parents[2])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "github_org_access_review.py",
            "--org",
            "onnx",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit):
        access_review.parse_args()


def test_output_dir_outside_repository_is_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["github_org_access_review.py", "--org", "onnx", "--output-dir", str(tmp_path)],
    )

    assert access_review.parse_args().output_dir == tmp_path.resolve()
