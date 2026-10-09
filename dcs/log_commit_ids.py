"""Commit metadata for telemetry log headers."""

import subprocess
from pathlib import Path


def commit_header():
    dcs_path = Path(__file__).resolve().parents[1]
    repositories = (
        ("dcs", dcs_path),
        ("asgard-alignment", dcs_path.parent / "asgard-alignment"),
    )
    commit_ids = []
    for name, path in repositories:
        try:
            commit_id = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot read {name} commit ID from {path}") from exc
        commit_ids.append(f"# {name} commit: {commit_id}\n")
    return "".join(commit_ids)
