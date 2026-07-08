"""Submission CSV writer + validator, matching the AIcrowd format.

Format (main track)::

    team_info,my cool team name,contact@email.com

    pid, track_uri_1, track_uri_2, ... track_uri_500
    pid, ...

Rules enforced by :func:`validate_submission`:
  * first non-blank line is ``team_info,<team>,<email>``;
  * each prediction row starts with an integer pid;
  * up to 500 track_uris per row, no duplicates, none from the seed;
  * every track_uri looks like ``spotify:track:<id>``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


def write_submission(
    path: str,
    predictions: Mapping[int, Sequence[str]],
    team_name: str,
    contact: str,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team_info", team_name, contact])
        fh.write("\n")
        for pid, uris in predictions.items():
            row = [pid] + list(uris)[:500]
            writer.writerow(row)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def validate_submission(
    path: str,
    seeds: Mapping[int, Iterable[str]] | None = None,
) -> ValidationResult:
    """Validate a submission file; optionally check no seed leaks per pid."""
    errors: List[str] = []
    seeds = {int(k): set(v) for k, v in (seeds or {}).items()}
    saw_team = False
    seen_pids = set()

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for lineno, row in enumerate(reader, start=1):
            if not row or all(c.strip() == "" for c in row):
                continue
            if not saw_team:
                if row[0].strip() != "team_info" or len(row) < 3:
                    errors.append(
                        f"line {lineno}: first non-blank row must be "
                        f"'team_info,<team>,<email>'"
                    )
                elif "@" not in row[2]:
                    errors.append(f"line {lineno}: contact must be an email")
                saw_team = True
                continue
            # prediction row
            try:
                pid = int(row[0])
            except ValueError:
                errors.append(f"line {lineno}: pid must be an integer, got {row[0]!r}")
                continue
            if pid in seen_pids:
                errors.append(f"line {lineno}: duplicate pid {pid}")
            seen_pids.add(pid)
            uris = [c.strip() for c in row[1:] if c.strip() != ""]
            if len(uris) > 500:
                errors.append(f"line {lineno}: pid {pid} has {len(uris)} > 500 tracks")
            if len(set(uris)) != len(uris):
                errors.append(f"line {lineno}: pid {pid} has duplicate track_uris")
            for u in uris:
                if not u.startswith("spotify:track:"):
                    errors.append(f"line {lineno}: bad track_uri {u!r}")
                    break
            leak = seeds.get(pid, set()).intersection(uris)
            if leak:
                errors.append(f"line {lineno}: pid {pid} leaks seed tracks {sorted(leak)[:3]}")

    if not saw_team:
        errors.append("missing team_info line")
    return ValidationResult(ok=(len(errors) == 0), errors=errors)
