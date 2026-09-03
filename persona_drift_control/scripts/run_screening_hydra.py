#!/usr/bin/env python3
"""Task-scoped Hydra entrypoint for the defended adversarial-screening
pipeline (same underlying code as scripts/run_defended_screening.py, which
keeps working unchanged -- existing sbatch scripts under environment/ call
that one directly and are unaffected by this file).

Only `task=defense` has a working `task.screening.*` schema: persona_drift
and sycophancy leave `task.screening` unset (null) in their conf/task/*.yaml
because screening.py::_make_controller (the entrypoint that actually drives
those two domains) doesn't support koopman_mpc yet -- see
docs/method/controllers.md "扩展点". Running this script with those tasks
raises a clear error instead of silently doing nothing meaningful.

See scripts/fit_koopman_hydra.py's docstring for why parameters come from
conf/task/<name>.yaml instead of argparse flags.

Usage:
    python scripts/run_screening_hydra.py output_dir=outputs/my_run \\
        task.screening.controller=koopman_mpc
    python scripts/run_screening_hydra.py output_dir=outputs/my_run \\
        task.screening.controller=threshold task.screening.threshold_y_min=0.6

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available -- see environment/setup_env.sh.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from persona_drift.adversarial_screening import run_adversarial_screening  # noqa: E402
from persona_drift.attack_trajectory import AttackTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.controller_cli import (  # noqa: E402
    load_koopman_mpc_controller,
    load_koopman_mpc_interaction_controller,
    make_controller_factory,
)

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="screening")
def main(cfg: DictConfig) -> None:
    if cfg.task.screening is None:
        raise NotImplementedError(
            f"task={cfg.task.name!r} has no task.screening.* config yet -- its controller "
            "factory (screening.py::_make_controller) doesn't support koopman_mpc, so there is "
            "nothing for this script to drive. See docs/method/controllers.md '扩展点'. "
            "Use task=defense, or wire up the persona-drift/sycophancy controller factory first."
        )
    s = cfg.task.screening
    output_dir = pathlib.Path(cfg.output_dir)

    judge_model = s.judge_model or s.agent_model
    trajectory_config = AttackTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=s.agent_max_new_tokens),
    )
    koopman_mpc_controller = (
        load_koopman_mpc_controller(
            pathlib.Path(s.koopman_model_path),
            s.koopman_model_key,
            s.koopman_nu,
            s.koopman_mu,
            s.koopman_horizon,
            s.koopman_repeat_penalty,
            contemporaneous_v=s.koopman_contemporaneous_v,
        )
        if s.controller == "koopman_mpc"
        else None
    )
    koopman_mpc_interaction_controller = (
        load_koopman_mpc_interaction_controller(
            pathlib.Path(s.koopman_interaction_model_path),
            s.koopman_nu,
            s.koopman_mu,
            s.koopman_horizon,
            s.koopman_repeat_penalty,
            contemporaneous_v=s.koopman_contemporaneous_v,
        )
        if s.controller == "koopman_mpc_interaction"
        else None
    )
    controller_factory = make_controller_factory(
        s.controller,
        s.threshold_y_min,
        koopman_mpc_controller,
        random_excite_p=s.random_excite_p,
        periodic_period=s.periodic_period,
        koopman_mpc_interaction_controller=koopman_mpc_interaction_controller,
    )
    report = run_adversarial_screening(
        agent_model_id=s.agent_model,
        judge_model_id=judge_model,
        output_dir=output_dir,
        num_attacks=s.num_attacks,
        seeds=tuple(s.seeds),
        attack_rng_seed=s.attack_rng_seed,
        device=s.device,
        trajectory_config=trajectory_config,
        controller_factory=controller_factory,
        attack_ids=list(s.attack_ids) if s.attack_ids else None,
    )
    print(f"task={cfg.task.name} controller={s.controller}")
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"report written to {output_dir}/adversarial_screening_report.md")


if __name__ == "__main__":
    main()
