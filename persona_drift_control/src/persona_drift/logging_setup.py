"""loguru-based structured logging for experiment runs.

Every run gets its own file under `logs/` (gitignored -- distinct from
`environment/slurm_logs/`'s raw sbatch stdout/stderr capture, which only
exists per Slurm job and is keyed by job ID, not by experiment config) named
by a run id, with the run's exact configuration logged as the very first
line so the log file alone is enough to know what produced it, without
having to cross-reference a Slurm job ID back to a command line.
"""

from __future__ import annotations

import pathlib
from typing import Any

from loguru import logger

DEFAULT_LOGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "logs"


def configure_run_logger(
    run_id: str, config: dict[str, Any], logs_dir: pathlib.Path | None = None
) -> int:
    """Adds a file sink for this run to loguru's process-wide logger.
    Deliberately does not call `logger.remove()` first, so the default
    stderr sink (which Slurm redirects into `environment/slurm_logs/*.out`)
    keeps working alongside the new file -- callers get both a live view via
    `squeue`/`tail` on the Slurm log and a permanent, config-tagged copy
    under `logs/`.

    Returns the sink id; pass it to `logger.remove(sink_id)` if the caller
    wants to stop logging to this file before the process exits (not
    necessary for a script that just runs to completion)."""

    logs_dir = pathlib.Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        logs_dir / f"{run_id}.log",
        level="INFO",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    logger.info("run_id={} config={}", run_id, config)
    return sink_id
