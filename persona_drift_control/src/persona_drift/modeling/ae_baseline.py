"""Encoder-decoder Koopman baseline (docs/experiments/ae_baseline_plan.md):
adapts `koopman_ae.core.DeepAugmentedKoopmanAutoencoder`'s architecture --
non-linear encoder z -> xi, linear dynamics xi_(t+1) = K xi_t + B v_t + c,
non-linear decoder xi -> z_hat -- to persona-drift's open-loop
`ReducedStateConfig` z_t/v_t/y_t schema (`modeling.dataset`).

Deliberately NOT importing `koopman_ae.core` (same reasoning as
`modeling.koopman.controllability_diagnostics`, see
`docs/method/koopman_surrogate.md`'s "why not reuse core.py" section): the
two sub-projects stay independently installable, and the MLP/ridge helpers
duplicated here are small and dependency-free.

Unlike `modeling.lstm_baseline.LSTMSurrogate`, `AEKoopmanSurrogate.step`/
`readout` operate on the SAME raw `z` representation `KoopmanSurrogate` uses
(the linear dynamics only happen internally, in the encoded latent space),
so this model satisfies `modeling.evaluate.Predictor` directly and can reuse
`one_step_error`/`rollout_output_error` unmodified -- see the plan doc's
"why not write new eval code" section.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class AEKoopmanConfig:
    latent_dim: int = 4
    hidden_dim: int = 4
    num_layers: int = 1
    activation: str = "tanh"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 300
    dynamics_alpha: float = 1e-4
    random_state: int = 0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 1e-6

    def __post_init__(self) -> None:
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 when set")


def _activation(name: str) -> type[nn.Module]:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    if name not in mapping:
        raise ValueError(f"unsupported activation: {name}")
    return mapping[name]


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_dim
    act = _activation(activation)
    for _ in range(num_layers):
        layers.extend([nn.Linear(current, hidden_dim), act()])
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


def _ridge_solve(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    gram = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(gram, X.T @ Y)


class AEKoopmanSurrogate:
    """`Predictor` protocol (step/readout) over raw `z`, mirroring
    `koopman.KoopmanSurrogate` -- the extension point
    `docs/method/koopman_surrogate.md` left for a future baseline whose
    state representation stays in the original `ReducedStateConfig` space.

    `readout(z) = float(z[0])`, matching `core.py`'s
    `DeepAugmentedKoopmanAutoencoder.predict_y` (a verbatim slice of the raw
    state's first component, not a separately learned read-out layer) --
    valid only under `nu=1` (z[0] is then exactly y_t), same assumption
    `koopman.abs_sign_extra_features` and the LSTM baseline make.
    """

    def __init__(self, state_dim: int, config: AEKoopmanConfig | None = None):
        self.state_dim = int(state_dim)
        self.config = config or AEKoopmanConfig()
        torch.manual_seed(self.config.random_state)
        self.encoder = _make_mlp(
            self.state_dim, self.config.latent_dim, self.config.hidden_dim, self.config.num_layers, self.config.activation
        )
        self.decoder = _make_mlp(
            self.config.latent_dim, self.state_dim, self.config.hidden_dim, self.config.num_layers, self.config.activation
        )
        self.K: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.c: np.ndarray | None = None
        self.training_history_: list[dict[str, float]] = []

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.array(values, dtype=np.float32, copy=True))

    def encode(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        z_2d = z_arr.reshape(1, -1) if z_arr.ndim == 1 else z_arr
        with torch.no_grad():
            xi = self.encoder(self._tensor(z_2d))
        out = xi.numpy()
        return out[0] if z_arr.ndim == 1 else out

    def decode(self, xi: np.ndarray) -> np.ndarray:
        xi_arr = np.asarray(xi, dtype=float)
        xi_2d = xi_arr.reshape(1, -1) if xi_arr.ndim == 1 else xi_arr
        with torch.no_grad():
            z = self.decoder(self._tensor(xi_2d))
        out = z.numpy()
        return out[0] if xi_arr.ndim == 1 else out

    def fit(
        self,
        dataset: dict[str, np.ndarray],
        val_dataset: dict[str, np.ndarray] | None = None,
    ) -> "AEKoopmanSurrogate":
        """`dataset`: output of `modeling.dataset.build_identification_dataset`
        (same shape `KoopmanSurrogate.fit` takes). Stage 1 trains encoder/decoder
        on reconstruction loss alone (`core.py`'s `training_mode=
        "reconstruction_then_ridge"`); stage 2 fits `K`/`B`/`c` by closed-form
        ridge in the frozen latent space (`core.py::_fit_latent_ridge`,
        reimplemented here for the same independent-installability reason as
        `modeling.koopman.controllability_diagnostics`).

        `val_dataset` (same shape as `dataset`, a held-out split): when given
        together with `config.early_stopping_patience`, stage 1 stops as soon
        as `val_dataset`'s reconstruction loss stops improving for that many
        epochs (restoring the best-seen encoder/decoder), instead of always
        running the full fixed `num_epochs` -- avoids having to hand-pick an
        epoch count that may stop the network before it has actually
        converged (see docs/experiments/ae_baseline_plan.md's "明确的局限").
        Without `val_dataset`, behavior is unchanged: exactly `num_epochs`."""

        Z, V, Z_next = dataset["Z"], dataset["V"], dataset["Z_next"]
        if Z.shape[0] == 0:
            raise ValueError("empty identification dataset")

        Z_t = self._tensor(Z)
        early_stopping_enabled = self.config.early_stopping_patience is not None and val_dataset is not None
        Z_val_t = self._tensor(val_dataset["Z"]) if early_stopping_enabled else None
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state: dict[str, dict] | None = None

        params = [*self.encoder.parameters(), *self.decoder.parameters()]
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        mse = nn.MSELoss()
        epoch_completed = 0
        for epoch in range(self.config.num_epochs):
            xi = self.encoder(Z_t)
            z_rec = self.decoder(xi)
            loss = mse(z_rec, Z_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_completed = epoch + 1
            history_entry = {"epoch": float(epoch_completed), "reconstruction_loss": float(loss.item())}
            if early_stopping_enabled:
                with torch.no_grad():
                    val_loss = float(mse(self.decoder(self.encoder(Z_val_t)), Z_val_t).item())
                history_entry["val_reconstruction_loss"] = val_loss
                if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch_completed
                    epochs_without_improvement = 0
                    best_state = {
                        "encoder": copy.deepcopy(self.encoder.state_dict()),
                        "decoder": copy.deepcopy(self.decoder.state_dict()),
                    }
                else:
                    epochs_without_improvement += 1
            self.training_history_.append(history_entry)
            if early_stopping_enabled and epochs_without_improvement >= self.config.early_stopping_patience:
                self.training_history_[-1]["early_stopped"] = True
                self.training_history_[-1]["restored_best_epoch"] = best_epoch
                self.training_history_[-1]["restored_best_val_loss"] = best_val_loss
                break
        if early_stopping_enabled and best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.decoder.load_state_dict(best_state["decoder"])
            if not self.training_history_[-1].get("early_stopped"):
                self.training_history_[-1]["early_stopped"] = False
                self.training_history_[-1]["restored_best_epoch"] = best_epoch
                self.training_history_[-1]["restored_best_val_loss"] = best_val_loss

        xi_t = self.encode(Z)
        xi_next = self.encode(Z_next)
        V = np.asarray(V, dtype=float)
        n = xi_t.shape[0]
        X = np.hstack([xi_t, V, np.ones((n, 1))])
        theta = _ridge_solve(X, xi_next, self.config.dynamics_alpha)
        latent_dim = self.config.latent_dim
        v_dim = V.shape[1]
        self.K = theta[:latent_dim].T
        self.B = theta[latent_dim : latent_dim + v_dim].T
        self.c = theta[latent_dim + v_dim :].reshape(-1)
        return self

    def _check_fit(self) -> None:
        if self.K is None:
            raise RuntimeError("fit() must be called before step()/readout()")

    def step(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predict z_(t+1) from (z_t, v_t) -- `Predictor.step`: encode ->
        linear latent step -> decode, all the way back to raw z-space, so
        the returned vector can be fed straight back into `step`/`readout`
        (or into `evaluate.rollout_output_error`'s loop) exactly like
        `KoopmanSurrogate.step`'s output."""

        self._check_fit()
        xi = self.encode(z)
        xi_next = self.K @ xi + self.B @ np.asarray(v, dtype=float) + self.c
        return self.decode(xi_next)

    def readout(self, z: np.ndarray) -> float:
        """Predict y_t from z_t -- `Predictor.readout`. See class docstring
        for why this is a verbatim slice rather than a learned read-out."""

        return float(np.asarray(z, dtype=float)[0])

    def n_params(self) -> int:
        encoder_decoder = sum(p.numel() for p in [*self.encoder.parameters(), *self.decoder.parameters()])
        dynamics = 0 if self.K is None else self.K.size + self.B.size + self.c.size
        return int(encoder_decoder) + int(dynamics)
