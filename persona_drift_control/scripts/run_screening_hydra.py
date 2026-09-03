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

Named arms live in conf/experiment/*.yaml, which pin an arm's output_dir
together with its controller settings; `persona_drift.run_config_guard` then makes
re-using another arm's directory an error rather than a silent mix of two
controllers' trajectories (the screening loop is resumable by design).

Usage:
    python scripts/run_screening_hydra.py experiment=phaseJ_budget1_koopman
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
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from persona_drift.adversarial_screening import run_adversarial_screening  # noqa: E402
from persona_drift.attack_trajectory import AttackTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.controller_cli import (  # noqa: E402
    load_koopman_mpc_controller,
    load_koopman_mpc_interaction_controller,
    make_controller_factory,
)
from persona_drift.run_config_guard import guard_run_config  # noqa: E402

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
    # Guard BEFORE anything is written or any model is loaded: the screening
    # loop resumes whatever complete trajectories it finds here, so a stale
    # output_dir would otherwise merge two arms into one report
    # (persona_drift.run_config_guard explains why Hydra's own .hydra/
    # snapshot doesn't catch this).
    changes = guard_run_config(
        output_dir, OmegaConf.to_container(cfg, resolve=True), allow_config_change=cfg.allow_config_change
    )
    if changes:
        print("allow_config_change=true: continuing {} despite config changes:\n{}".format(
            output_dir, "\n".join(changes)
        ))

    judge_model = s.judge_model or s.agent_model
    trajectory_config = AttackTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=s.agent_max_new_tokens),
    )
    # `name=s.controller` (not the loaders' "koopman_mpc" default): the
    # interaction variant used to record itself as plain `koopman_mpc` in the
    # `excitation_design` column and the run id, which is harmless when one
    # directory holds one arm but useless once several budgeted/unbudgeted
    # variants are compared. The loader appends the budget suffix itself
    # (controller_cli._budgeted_name), so unbudgeted runs through the argparse
    # CLI keep their historical names untouched.
    koopman_mpc_controller = (
        load_koopman_mpc_controller(
            pathlib.Path(s.koopman_model_path),
            s.koopman_model_key,
            s.koopman_nu,
            s.koopman_mu,
            s.koopman_horizon,
            s.koopman_repeat_penalty,
            contemporaneous_v=s.koopman_contemporaneous_v,
            pad_short_history=s.koopman_pad_short_history,
            remind_budget=s.remind_budget,
            episode_length=s.episode_length,
            name=s.controller,
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
            pad_short_history=s.koopman_pad_short_history,
            remind_budget=s.remind_budget,
            episode_length=s.episode_length,
            name=s.controller,
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
        fixed_schedule_turns=tuple(s.fixed_schedule_turns) if s.fixed_schedule_turns else None,
        remind_budget=s.remind_budget,
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
