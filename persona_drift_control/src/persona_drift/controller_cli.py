"""Shared controller-construction CLI helpers for
scripts/run_defended_screening.py and scripts/run_benign_helpfulness_screening.py:
both scripts expose the same --controller {zero_control,constant_remind,
threshold,periodic,koopman_mpc[,random_excite]} choice and need to (a) load a fitted
Koopman surrogate from a koopman_fit_report.json when --controller
koopman_mpc, and (b) turn the parsed args into a controller_factory for
adversarial_screening.run_adversarial_screening / benign_screening.run_benign_screening.
random_excite is only meaningful for run_defended_screening.py's
open-loop-excitation phase, so it's an optional branch here rather than a
hard requirement.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Callable

from .control import (
    ConstantRemindController,
    Controller,
    KoopmanMPCController,
    PeriodicController,
    RandomExciteController,
    ThresholdController,
    ZeroControlController,
)
from .modeling.dataset import ReducedStateConfig
from .modeling.interaction_lift import InteractionLiftedSurrogate
from .modeling.koopman import abs_sign_extra_features, no_extra_features, surrogate_from_arrays

EXTRA_FEATURES_FNS = {"arx": no_extra_features, "richer_abs_sign": abs_sign_extra_features}


def load_koopman_mpc_controller(
    model_path: pathlib.Path,
    model_key: str,
    nu: int,
    mu: int,
    horizon: int,
    repeat_penalty: float,
    contemporaneous_v: bool = False,
) -> KoopmanMPCController:
    report = json.loads(model_path.read_text())
    fit = report[model_key]
    state_config = ReducedStateConfig(nu=nu, mu=mu, contemporaneous_v=contemporaneous_v)
    surrogate = surrogate_from_arrays(
        fit["A"],
        fit["B"],
        fit["b"],
        fit["C"],
        state_dim=state_config.state_dim,
        extra_features_fn=EXTRA_FEATURES_FNS[model_key],
    )
    return KoopmanMPCController(
        surrogate=surrogate,
        state_config=state_config,
        horizon=horizon,
        repeat_penalty=repeat_penalty,
    )


def load_koopman_mpc_interaction_controller(
    model_path: pathlib.Path,
    nu: int,
    mu: int,
    horizon: int,
    repeat_penalty: float,
    contemporaneous_v: bool = False,
) -> KoopmanMPCController:
    """Phase H (docs/experiments/koopman_case_study_design.md's "后续:
    验证'对下一步方向的启示'"): loads a state-action-interaction-augmented
    surrogate (fit by scripts/analyze_state_action_interaction.py, saved
    under its "model" key -- A/B/b/C where B has 2 columns, `[v, v*y_t]`)
    and wraps it in `InteractionLiftedSurrogate` so `KoopmanMPCController`
    drives it through the same unmodified `Predictor.step(z, v)` (`v`
    still a raw length-1 action) as every other koopman_mpc variant. Always
    `no_extra_features` (plain ARX) lifting, never `abs_sign` -- see the doc
    for why `abs_sign_extra_features` is near-degenerate on this bounded
    [0,1] safety score and was deliberately not used for this model.

    `contemporaneous_v` must match whatever `ReducedStateConfig` the saved
    model was fit with (`analyze_state_action_interaction.py --contemporaneous-v`
    for the corrected v-alignment, docs/next step.md 2026-09-02) -- a
    mismatch here silently reintroduces the alignment bug at inference
    time even with a correctly-fit model."""

    report = json.loads(model_path.read_text())
    fit = report["model"]
    state_config = ReducedStateConfig(nu=nu, mu=mu, contemporaneous_v=contemporaneous_v)
    surrogate = surrogate_from_arrays(
        fit["A"], fit["B"], fit["b"], fit["C"], state_dim=state_config.state_dim, extra_features_fn=no_extra_features
    )
    wrapped = InteractionLiftedSurrogate(surrogate=surrogate, state_index=0)
    return KoopmanMPCController(
        surrogate=wrapped,
        state_config=state_config,
        horizon=horizon,
        repeat_penalty=repeat_penalty,
    )


def _excitation_seed(seed: int, entry_id: str) -> int:
    """Derives a per-(entry, seed) RNG seed for `RandomExciteController`.

    Found while executing docs/next step.md (2026-09-02): every caller of
    this factory used to build `RandomExciteController(seed=seed)` with
    `seed` alone (just the trajectory-level 0/1 from `--seeds`), and
    `run_trajectories_loop` calls this factory fresh for every `(entry,
    seed)` pair -- so `random.Random(seed)` produced a BYTE-IDENTICAL
    5-turn `u_remind` draw for every entry sharing the same `seed` value.
    Confirmed directly in outputs/koopman_defense_phaseB_random_excite/
    trajectories.jsonl: all 30 `seed0` attacks got the exact same sequence,
    all 30 `seed1` attacks got a different but also-shared-across-attacks
    sequence -- only 2 independent excitation realizations in the whole
    300-row dataset, not the "i.i.d. Bernoulli(p) each turn" the class
    docstring promises. `scripts/analyze_phaseB_seed_confound.py` quantifies
    how much this cost (turns out: not enough to invalidate that dataset's
    B estimate, but it did widen its CI to include zero).

    Uses hashlib (not the builtin `hash()`, which is randomized per-process
    for str/bytes by default via PYTHONHASHSEED) so this stays deterministic
    and reproducible across runs -- required for `run_trajectories_loop`'s
    resumability: re-deriving the seed for the same `(entry_id, seed)` after
    a crash must reproduce the exact same draws as an uninterrupted run."""

    digest = hashlib.sha256(f"{entry_id}::seed{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_controller_factory(
    name: str,
    threshold_y_min: float,
    koopman_mpc_controller: KoopmanMPCController | None,
    random_excite_p: float | None = None,
    periodic_period: int | None = None,
    koopman_mpc_interaction_controller: KoopmanMPCController | None = None,
) -> Callable[[int, str], Controller]:
    """Returns a `(seed, entry_id) -> Controller` factory -- `entry_id`
    (the attack_id/benign_id the trajectory is being built for, see
    `screening_common.run_trajectories_loop`) defaults to `""` so every
    caller that doesn't need it (every branch here except `random_excite`)
    can keep calling `factory(seed)` unchanged."""

    if name == "zero_control":
        return lambda seed, entry_id="": ZeroControlController()
    if name == "constant_remind":
        return lambda seed, entry_id="": ConstantRemindController()
    if name == "threshold":
        return lambda seed, entry_id="": ThresholdController(y_min=threshold_y_min)
    if name == "periodic":
        if periodic_period is None:
            raise ValueError("periodic_period is required for --controller periodic")
        return lambda seed, entry_id="": PeriodicController(period=periodic_period)
    if name == "random_excite":
        if random_excite_p is None:
            raise ValueError("random_excite_p is required for --controller random_excite")
        return lambda seed, entry_id="": RandomExciteController(p=random_excite_p, seed=_excitation_seed(seed, entry_id))
    if name == "koopman_mpc":
        # Stateless given a fixed fitted surrogate -- safe to hand out the
        # same instance to every trajectory (unlike RandomExciteController,
        # there's no per-trajectory RNG state to keep independent).
        return lambda seed, entry_id="": koopman_mpc_controller
    if name == "koopman_mpc_interaction":
        return lambda seed, entry_id="": koopman_mpc_interaction_controller
    raise ValueError(f"unknown controller: {name!r}")
