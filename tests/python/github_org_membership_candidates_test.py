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


@pytest.mark.parametrize("item_type", ["issue", "pr"])
def test_search_excludes_drafts_only_for_pull_requests(
    monkeypatch: pytest.MonkeyPatch, item_type: str
) -> None:
    queries = []

    def run_gh(arguments: list[str]) -> dict:
        queries.append(parse_qs(urlsplit(arguments[1]).query)["q"][0])
        return {"items": []}

    monkeypatch.setattr(membership_candidates, "run_gh", run_gh)

    membership_candidates.search_issues_or_pulls(
        "onnx", item_type, date(2026, 1, 1), date(2026, 10, 1)
    )

    assert queries == [
        f"org:onnx is:{item_type} created:2026-01-01..2026-10-01"
        + (" draft:false" if item_type == "pr" else "")
    ]


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
