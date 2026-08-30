"""Loads the offline-generated scripted-user-turn library (see
docs/SCRIPTED_USER_TURNS_FEASIBILITY.md and scripts/generate_user_scripts.py)
for `TrajectoryConfig.user_mode == "scripted"`. Analysis-only until this
module existed -- see the feasibility doc's "结论摘要" for why this isn't the
default and needs a live-vs-scripted consistency check before being trusted.
"""

from __future__ import annotations

import json
import pathlib
import re

USER_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "resources" / "user_scripts"


def slugify_topic(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")


def load_user_script(topic: str, seed: int, scripts_dir: pathlib.Path | None = None) -> list[str]:
    scripts_dir = scripts_dir or USER_SCRIPTS_DIR
    path = scripts_dir / f"{slugify_topic(topic)}__seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no scripted user turns at {path} -- run scripts/generate_user_scripts.py first "
            "with a --seeds list covering every seed this trajectory needs"
        )
    return json.loads(path.read_text())
