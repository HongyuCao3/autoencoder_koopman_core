"""LSTM surrogate baseline (docs/experiments/lstm_baseline_plan.md, design
option B): unlike `koopman.KoopmanSurrogate`'s fixed `(nu, mu)` window, this
model's hidden state `(h, c)` evolves continuously across an entire
trajectory, so it can in principle learn to remember more (or less) than the
`nu=1, mu=2` window `richer_abs_sign` was restricted to -- the actual
question `BASELINES.md`'s layer-3 ablation asks: does the Koopman surrogate's
finite-memory linear structure leave learnable signal on the table relative
to a real non-linear recurrent model?

Two distinct evaluation regimes, mirroring `modeling.evaluate`'s one-step /
rollout pair but NOT reusing that module's functions directly (they assume
`ReducedStateConfig`-shaped z vectors, which do not apply to an opaque
`(h, c)` state -- see the module docstring in `evaluate.py` and
`docs/experiments/lstm_baseline_plan.md`):

- teacher-forced ("one-step"): `forward_trajectory` re-grounds on the TRUE
  `y_t` at every step, comparable to `KoopmanSurrogate`'s `train_one_step_mse`.
- free rollout: `step`/`readout` (the `Predictor` protocol) chain the model's
  OWN prediction back in as the next input, comparable to
  `rollout_output_error`/`held_out_rollout_mse`. Since the true `y_t` is not
  used as an input once free rollout starts, `step` uses `readout(z)` (the
  model's own current belief) as the pseudo-observation fed alongside `v_t`
  -- the recurrent-model analogue of Koopman's `step()` propagating its own
  linear state estimate rather than re-observing truth.

`contemporaneous_v` (added 2026-09-03) is this module's counterpart of
`dataset.ReducedStateConfig.contemporaneous_v`: with it on, the input paired
with `y_t` to predict `y_(t+1)` is `v_(t+1)` -- the reminder inserted BEFORE
the reply that produces `y_(t+1)`, i.e. the one that actually causes it --
instead of `v_t`, whose effect on `y_(t+1)` is only the carryover of a
reminder already scored one turn earlier. It is the same one-slot shift, on
the same data and for the same reason, as the Koopman side's fix; the
original 2026-09-01 LSTM numbers were produced under the default (`False`),
so any comparison against a v-aligned Koopman baseline has to pass `True`.
See docs/experiments/lstm_baseline_plan.md and
docs/experiments/koopman_case_study_design.md's Phase I.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMSurrogate(nn.Module):
    """Single `nn.LSTMCell(input_size=2, hidden_size=hidden_size)` reading
    `[y_t, v_t]` each turn, plus a linear readout `Linear(hidden_size, 1)`
    playing the role of `KoopmanSurrogate.C`. Trajectories start from an
    all-zero `(h, c)` (no prior knowledge assumed, the standard `LSTMCell`
    convention)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size=2, hidden_size=hidden_size)
        self.readout_layer = nn.Linear(hidden_size, 1)

    def init_state(self) -> np.ndarray:
        return np.zeros(2 * self.hidden_size, dtype=float)

    def _unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float)
        return z[: self.hidden_size], z[self.hidden_size :]

    def forward_trajectory(
        self, ys: torch.Tensor, vs: torch.Tensor, contemporaneous_v: bool = False
    ) -> torch.Tensor:
        """Teacher-forced forward pass over one trajectory (`ys`/`vs`: 1D
        tensors of length T, the trajectory's true `y_t`/`v_t`). Returns a
        length-T tensor of predicted `y_t` for t=0..T-1: `preds[0]` is the
        readout of the all-zero initial state (the model's unconditional
        prior, before any input is seen); `preds[t]` for t>=1 is the readout
        AFTER the cell has processed the true `(y_(t-1), v_(t-1))` -- i.e.
        re-grounded on truth at every step, never on its own prediction.

        With `contemporaneous_v=True` the cell reads `(y_(t-1), v_t)`
        instead, so the action fed in is the one that acts on the turn being
        predicted (see the module docstring)."""

        shift = 1 if contemporaneous_v else 0
        h = torch.zeros(1, self.hidden_size)
        c = torch.zeros(1, self.hidden_size)
        preds = [self.readout_layer(h).reshape(())]
        for t in range(ys.shape[0] - 1):
            inp = torch.stack([ys[t], vs[t + shift]]).reshape(1, 2)
            h, c = self.cell(inp, (h, c))
            preds.append(self.readout_layer(h).reshape(()))
        return torch.stack(preds)

    def step(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """`Predictor.step`: advances `(h, c)` using `readout(z)` (the
        model's own current y-estimate, since free rollout never observes
        the true y again after the first turn) and `v` as the cell input."""

        h, c = self._unpack(z)
        y_self = self.readout(z)
        v_val = float(np.asarray(v, dtype=float).reshape(-1)[0])
        with torch.no_grad():
            inp = torch.tensor([[y_self, v_val]], dtype=torch.float32)
            h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
            c_t = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
            h_next, c_next = self.cell(inp, (h_t, c_t))
        return np.concatenate([h_next.squeeze(0).numpy(), c_next.squeeze(0).numpy()])

    def readout(self, z: np.ndarray) -> float:
        """`Predictor.readout`: linear map from the `h` half of `z`."""

        h, _ = self._unpack(z)
        with torch.no_grad():
            val = self.readout_layer(torch.tensor(h, dtype=torch.float32).unsqueeze(0))
        return float(val.item())


def teacher_forced_predictions(
    model: LSTMSurrogate,
    traj_rows: list[dict],
    y_col: str = "y_safety",
    u_col: str = "u_remind",
    contemporaneous_v: bool = False,
) -> list[tuple[int, float, float]]:
    """One trajectory's rows (sorted by turn) -> `(turn_index, y_true,
    y_pred)` for turn_index=0..T-1, teacher forcing every step. `turn_index`
    is a 0-based position within the trajectory (not the raw `turn` column),
    so callers can slice by position to match `ReducedStateConfig`'s
    `start = max(nu-1, mu)` cutoff for an apples-to-apples comparison against
    a Koopman model that cannot produce a prediction before that position."""

    ys = [float(row[y_col]) for row in traj_rows]
    vs = [float(row[u_col]) for row in traj_rows]
    with torch.no_grad():
        preds = model.forward_trajectory(
            torch.tensor(ys, dtype=torch.float32),
            torch.tensor(vs, dtype=torch.float32),
            contemporaneous_v=contemporaneous_v,
        )
    return list(zip(range(len(ys)), ys, preds.numpy().tolist()))


def rollout_predictions(
    model: LSTMSurrogate,
    traj_rows: list[dict],
    y_col: str = "y_safety",
    u_col: str = "u_remind",
    contemporaneous_v: bool = False,
) -> list[tuple[int, float, float]]:
    """One trajectory's rows -> `(turn_index, y_true, y_pred)` for
    turn_index=0..T-1, free rollout: `z` starts at `init_state()` (all-zero,
    NOT seeded from the true y0 -- `(h, c)` cannot be constructed from a raw
    y-value the way a `ReducedStateConfig` window can), then only `v_t` is
    fed in from the true trajectory; `y_t` predictions never re-ground on
    truth after the start. `contemporaneous_v` shifts which `v` that is,
    exactly as in `forward_trajectory`."""

    shift = 1 if contemporaneous_v else 0
    ys = [float(row[y_col]) for row in traj_rows]
    vs = [float(row[u_col]) for row in traj_rows]
    z = model.init_state()
    preds = [model.readout(z)]
    for t in range(len(ys) - 1):
        z = model.step(z, np.array([vs[t + shift]]))
        preds.append(model.readout(z))
    return list(zip(range(len(ys)), ys, preds))


def mse_from_predictions(predictions_by_traj: list[list[tuple[int, float, float]]], min_turn_index: int = 0) -> float:
    """Mean squared error over `(turn_index, y_true, y_pred)` triples across
    trajectories, restricted to `turn_index >= min_turn_index` -- pass
    `min_turn_index=max(nu-1, mu)` to compare against a Koopman model fit
    with that `ReducedStateConfig` on exactly the same set of turns."""

    errors = [
        (y_pred - y_true) ** 2
        for preds in predictions_by_traj
        for turn_index, y_true, y_pred in preds
        if turn_index >= min_turn_index
    ]
    return float(np.mean(errors)) if errors else float("nan")


def train_lstm_surrogate(
    hidden_size: int,
    train_rows_by_traj: list[list[dict]],
    early_stop_rows_by_traj: list[list[dict]],
    y_col: str = "y_safety",
    u_col: str = "u_remind",
    epochs: int = 200,
    lr: float = 1e-2,
    patience: int = 20,
    seed: int = 0,
    min_turn_index: int = 0,
    contemporaneous_v: bool = False,
) -> tuple[LSTMSurrogate, dict]:
    """Trains by teacher-forced BPTT (loss = per-turn MSE against the true
    `y_t`, summed over an entire trajectory before each optimizer step --
    matches `KoopmanSurrogate.fit`'s closed-form objective's spirit: fit
    everything from `nu, mu` in one shot, just via gradient descent instead
    of ridge regression since `LSTMCell` has no closed form). Early stops on
    `early_stop_rows_by_traj`'s ROLLOUT mse (`min_turn_index`-matched to
    whatever Koopman window this run is being compared against), not train
    loss, to avoid reporting an LSTM that is still improving training loss by
    overfitting the small identification set.

    `early_stop_rows_by_traj` is SELECTED ON -- the returned model is the
    best epoch by this set's rollout MSE -- so passing the same trajectories
    that will later be reported as held-out makes that reported number
    optimistic. The 2026-09-01 run did exactly that (and still lost to the
    linear baseline, so it did not matter then); once the two are close, the
    caller has to carve this set out of its training attacks instead. See
    fit_koopman_lstm_baseline.py's --val-frac and
    docs/experiments/lstm_baseline_plan.md."""

    torch.manual_seed(seed)
    model = LSTMSurrogate(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensors = [
        (
            torch.tensor([float(r[y_col]) for r in rows], dtype=torch.float32),
            torch.tensor([float(r[u_col]) for r in rows], dtype=torch.float32),
        )
        for rows in train_rows_by_traj
    ]

    best_rollout_mse = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = torch.tensor(0.0)
        for ys, vs in train_tensors:
            preds = model.forward_trajectory(ys, vs, contemporaneous_v=contemporaneous_v)
            total_loss = total_loss + torch.mean((preds - ys) ** 2)
        loss = total_loss / max(len(train_tensors), 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        early_stop_preds = [
            rollout_predictions(model, rows, y_col=y_col, u_col=u_col, contemporaneous_v=contemporaneous_v)
            for rows in early_stop_rows_by_traj
        ]
        early_stop_rollout_mse = mse_from_predictions(early_stop_preds, min_turn_index=min_turn_index)
        history.append(
            {"epoch": epoch, "train_loss": float(loss.item()), "early_stop_rollout_mse": early_stop_rollout_mse}
        )

        if early_stop_rollout_mse < best_rollout_mse:
            best_rollout_mse = early_stop_rollout_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": history[-1 - epochs_without_improvement]["epoch"], "n_epochs_run": len(history)}
