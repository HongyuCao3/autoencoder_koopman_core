"""Shared controller-construction CLI helpers for
scripts/run_defended_screening.py and scripts/run_benign_helpfulness_screening.py:
both scripts expose the same --controller {zero_control,constant_remind,
threshold,koopman_mpc[,random_excite]} choice and need to (a) load a fitted
Koopman surrogate from a koopman_fit_report.json when --controller
koopman_mpc, and (b) turn the parsed args into a controller_factory for
adversarial_screening.run_adversarial_screening / benign_screening.run_benign_screening.
random_excite is only meaningful for run_defended_screening.py's
open-loop-excitation phase, so it's an optional branch here rather than a
hard requirement.
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

from .control import (
    ConstantRemindController,
    Controller,
    KoopmanMPCController,
    RandomExciteController,
    ThresholdController,
    ZeroControlController,
)
from .modeling.dataset import ReducedStateConfig
from .modeling.koopman import abs_sign_extra_features, no_extra_features, surrogate_from_arrays

EXTRA_FEATURES_FNS = {"arx": no_extra_features, "richer_abs_sign": abs_sign_extra_features}


def load_koopman_mpc_controller(
    model_path: pathlib.Path,
    model_key: str,
    nu: int,
    mu: int,
    horizon: int,
    repeat_penalty: float,
) -> KoopmanMPCController:
    report = json.loads(model_path.read_text())
    fit = report[model_key]
    surrogate = surrogate_from_arrays(
        fit["A"],
        fit["B"],
        fit["b"],
        fit["C"],
        state_dim=ReducedStateConfig(nu=nu, mu=mu).state_dim,
        extra_features_fn=EXTRA_FEATURES_FNS[model_key],
    )
    return KoopmanMPCController(
        surrogate=surrogate,
        state_config=ReducedStateConfig(nu=nu, mu=mu),
        horizon=horizon,
        repeat_penalty=repeat_penalty,
    )


def make_controller_factory(
    name: str,
    threshold_y_min: float,
    koopman_mpc_controller: KoopmanMPCController | None,
    random_excite_p: float | None = None,
) -> Callable[[int], Controller]:
    if name == "zero_control":
        return lambda seed: ZeroControlController()
    if name == "constant_remind":
        return lambda seed: ConstantRemindController()
    if name == "threshold":
        return lambda seed: ThresholdController(y_min=threshold_y_min)
    if name == "random_excite":
        if random_excite_p is None:
            raise ValueError("random_excite_p is required for --controller random_excite")
        return lambda seed: RandomExciteController(p=random_excite_p, seed=seed)
    if name == "koopman_mpc":
        # Stateless given a fixed fitted surrogate -- safe to hand out the
        # same instance to every trajectory (unlike RandomExciteController,
        # there's no per-trajectory RNG state to keep independent).
        return lambda seed: koopman_mpc_controller
    raise ValueError(f"unknown controller: {name!r}")
