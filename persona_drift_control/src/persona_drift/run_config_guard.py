"""Refuses to continue a run into an output directory that already holds a
run with a different config.

Why this exists: the screening loops are deliberately resumable
(`screening_common.prepare_resumable_trajectories_file` re-uses whatever
complete trajectories it finds), which is exactly what makes a stale
`output_dir` dangerous -- pointing a new arm at an old arm's directory does
not fail, it silently mixes trajectories produced by two different
controllers into one report. That risk was acceptable while every run was
one hand-reviewed sbatch file with one `--output-dir`; it stops being
acceptable once arms are composed from Hydra overrides, where changing a
controller but not the output directory is a one-token mistake (see
conf/experiment/*.yaml, docs/experiments/budget_constrained_defense_plan.md).

Hydra's own `.hydra/config.yaml` snapshot does not cover this: it writes a
new timestamped run directory each time, so nothing ever compares this run
against the previous one in the same output directory.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

RUN_CONFIG_FILENAME = "hydra_run_config.json"


def flatten_config(node: Any, prefix: str = "") -> dict[str, str]:
    """Flattens nested config dicts to `a.b.c -> repr(value)` so two configs
    can be diffed key by key in an error message (rather than dumping two
    whole YAML blobs and leaving the reader to spot the difference)."""

    if isinstance(node, dict):
        flat: dict[str, str] = {}
        for key, value in node.items():
            flat.update(flatten_config(value, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): repr(node)}


def describe_config_changes(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """`old != new` lines for every key that differs, added or removed."""

    old_flat, new_flat = flatten_config(previous), flatten_config(current)
    lines = []
    for key in sorted(set(old_flat) | set(new_flat)):
        old_value, new_value = old_flat.get(key, "<absent>"), new_flat.get(key, "<absent>")
        if old_value != new_value:
            lines.append(f"    {key}: {old_value} -> {new_value}")
    return lines


def guard_run_config(
    output_dir: pathlib.Path, snapshot: dict[str, Any], allow_config_change: bool = False
) -> list[str]:
    """Records `snapshot` in `output_dir` on the first run and compares
    against it afterwards. Returns the list of changed-key descriptions
    (empty when the config matched or the directory was fresh).

    Raises `SystemExit` on a mismatch unless `allow_config_change`, in which
    case the new snapshot replaces the old one and the changes are returned
    for the caller to log."""

    output_dir = pathlib.Path(output_dir)
    path = output_dir / RUN_CONFIG_FILENAME
    if not path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        return []

    previous = json.loads(path.read_text())
    if previous == snapshot:
        return []

    changes = describe_config_changes(previous, snapshot)
    if not allow_config_change:
        detail = "\n".join(changes)
        raise SystemExit(
            f"{path} records a different config for this output_dir:\n{detail}\n"
            "  Refusing to resume: this directory's existing trajectories were produced by the\n"
            "  recorded config, and resuming into it would produce one report averaging two arms.\n"
            "  Point output_dir at a new directory (conf/experiment/*.yaml pins one per arm), or\n"
            "  pass allow_config_change=true if the change really is meant to apply to this run."
        )
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return changes
