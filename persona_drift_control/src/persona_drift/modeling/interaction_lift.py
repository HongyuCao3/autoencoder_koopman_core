"""State-action interaction control channel (path 2 of "how to prove
Koopman's motivation", docs/experiments/koopman_case_study_design.md's "对
下一步方向的启示"): the case study proved that as long as the action `v`
enters the transition only through a flat, state-independent `B @ v`, the
marginal predicted value of remind-vs-not is provably constant in `z` --
`docs/experiments/lstm_baseline_plan.md`'s repeat-penalty sweep confirms
this quantitatively (every tested penalty produces either all-remind or
never-remind, never a mix, margin variance at floating-point noise). The
only way to make that marginal value depend on state is to give the action
itself a state-dependent component, i.e. augment the control input with an
explicit interaction term rather than adding more lifted STATE features
(`extra_features_fn` only adds columns to `psi`, which are still multiplied
by the same flat `B` acting on `v` alone -- that cannot create the needed
coupling, see the case study).

`KoopmanSurrogate.fit`/`step` already support `d_v > 1` (nothing here
touches `modeling/koopman.py`): fit on an augmented `V_aug = [v, v * y]`
(the second column literally couples the action to the current safety
reading) and `B` becomes `(d_psi, 2)`, so `C @ (B[:, 0] + B[:, 1] * y_t)` is
the marginal value of reminding -- now linear IN `y_t`, no longer constant.
`InteractionLiftedSurrogate` hides this augmentation behind the plain
`Predictor` protocol (`step(z, v)` with `v` a length-1 raw action, same as
every other predictor in this codebase) so `KoopmanMPCController`,
`modeling.evaluate.one_step_error`/`rollout_output_error` all drive it
completely unmodified -- see docs/experiments/koopman_case_study_design.md
for why this is deliberately a wrapper rather than a change to
`control.py`/`koopman.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .koopman import KoopmanSurrogate


def augment_with_interaction(V: np.ndarray, Z: np.ndarray, state_index: int = 0) -> np.ndarray:
    """`V`: (n, 1) raw action column (every controller in this codebase uses
    a single binary `u_remind`). Returns `(n, 2)`: the original column plus
    `v_t * z_t[state_index]` -- `state_index=0` is the current `y` reading
    for any `ReducedStateConfig(nu=1, ...)` (mirrors the same `nu==1`
    assumption `koopman.abs_sign_extra_features` already documents)."""

    interaction = V[:, :1] * Z[:, state_index : state_index + 1]
    return np.hstack([V, interaction])


@dataclass
class InteractionLiftedSurrogate:
    """`Predictor`-protocol wrapper around a `KoopmanSurrogate` fit with
    `augment_with_interaction`'s `V_aug`. Callers pass the same raw
    length-1 action `v` they would to a plain `KoopmanSurrogate`; this class
    lifts it to `[v, v * z[state_index]]` before delegating -- the
    augmentation never needs to be visible to `control.py` or
    `modeling.evaluate`."""

    surrogate: KoopmanSurrogate
    state_index: int = 0

    def step(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        v_raw = float(np.asarray(v, dtype=float).reshape(-1)[0])
        v_aug = np.array([v_raw, v_raw * z[self.state_index]])
        return self.surrogate.step(z, v_aug)

    def readout(self, z: np.ndarray) -> float:
        return self.surrogate.readout(z)
