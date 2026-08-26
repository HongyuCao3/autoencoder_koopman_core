from __future__ import annotations

import os
import pathlib
import random
import shutil
import signal
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without torch.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def _as_2d(values: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _optional_float(value: Any) -> float:
    return np.nan if value is None else float(value)


def _ridge_solve(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.pinv(X.T @ X + reg) @ X.T @ Y


def controllability_diagnostics(A: np.ndarray, B: np.ndarray, horizon: int) -> dict[str, Any]:
    """Finite-horizon controllability diagnostics for linear controlled dynamics."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    blocks = []
    gramian = np.zeros((A.shape[0], A.shape[0]))
    Ak = np.eye(A.shape[0])
    for _ in range(horizon):
        block = Ak @ B
        blocks.append(block)
        gramian += block @ block.T
        Ak = Ak @ A
    ctrb = np.concatenate(blocks, axis=1) if blocks else np.empty((A.shape[0], 0))
    singular_values = np.linalg.svd(ctrb, compute_uv=False) if ctrb.size else np.array([])
    gramian_eigs = np.linalg.eigvalsh(gramian)
    eigvals = np.linalg.eigvals(A)
    return {
        "controllability_rank": int(np.linalg.matrix_rank(ctrb)) if ctrb.size else 0,
        "controllability_matrix": ctrb.tolist(),
        "controllability_singular_values": singular_values.tolist(),
        "gramian": gramian.tolist(),
        "gramian_eigenvalues": gramian_eigs.tolist(),
        "gramian_condition": float(
            np.linalg.cond(gramian + 1e-12 * np.eye(A.shape[0]))
        ),
        "A_eigenvalues_real": [float(v.real) for v in eigvals],
        "A_eigenvalues_imag": [float(v.imag) for v in eigvals],
        "spectral_radius": float(max(abs(v) for v in eigvals)) if len(eigvals) else np.nan,
    }


@dataclass(frozen=True)
class AugmentedStateConfig:
    """Configuration for Model III augmented output-state construction."""

    output_memory: int = 2
    input_memory: int = 2
    control_mode: str = "error"
    output_columns: tuple[str, ...] = ("normalized_output",)
    target_columns: tuple[str, ...] = ("effective_norm",)

    def __post_init__(self) -> None:
        if self.output_memory < 1:
            raise ValueError("output_memory must be >= 1")
        if self.input_memory < 0:
            raise ValueError("input_memory must be >= 0")
        if self.control_mode not in {"error", "error_abs_sign"}:
            raise ValueError(f"unsupported control_mode: {self.control_mode}")


@dataclass
class AugmentedStateDataset:
    meta: pd.DataFrame
    Z_t: np.ndarray
    R: np.ndarray
    Z_next: np.ndarray
    output_dim: int
    control_dim: int
    target_dim: int
    output_memory: int
    input_memory: int
    control_mode: str

    @property
    def state_dim(self) -> int:
        return int(self.Z_t.shape[1])


def _control_from_y_r(y: np.ndarray, r: np.ndarray, control_mode: str) -> np.ndarray:
    if r.shape[0] == 1 and y.shape[0] != 1:
        r_for_error = np.repeat(r, y.shape[0])
    elif r.shape[0] == y.shape[0]:
        r_for_error = r
    elif r.shape[0] > y.shape[0]:
        # Allow exogenous target/context vectors such as [target, task_onehot...].
        # Only the output-aligned target components define the feedback error.
        r_for_error = r[: y.shape[0]]
    else:
        raise ValueError(
            f"control_mode={control_mode} requires scalar target or target_dim==output_dim; "
            f"got target_dim={r.shape[0]}, output_dim={y.shape[0]}"
        )
    error = r_for_error - y
    if control_mode == "error":
        return error
    return np.concatenate([error, np.abs(error), np.sign(error)])


def _state_at_turn(
    ys: dict[int, np.ndarray],
    us: dict[int, np.ndarray],
    turn: int,
    config: AugmentedStateConfig,
) -> np.ndarray:
    y_parts = [ys[turn - lag] for lag in range(config.output_memory)]
    u_parts = [us[turn - lag] for lag in range(config.input_memory)]
    return np.concatenate([*y_parts, *u_parts]).astype(float)


def build_augmented_state_dataset(
    trajectories: pd.DataFrame,
    config: AugmentedStateConfig | None = None,
) -> AugmentedStateDataset:
    """Convert rollout trajectories into Model III tuples `(z_t, r, z_(t+1))`.

    Model III treats the delay-embedded output/control history as a Markov
    reduced state:

        z_t = [y_t, y_(t-1), ..., u_t, u_(t-1), ...].

    If no explicit numerical control is logged, the default control is the
    tracking error `u_t = r - y_t`.
    """

    cfg = config or AugmentedStateConfig()
    meta_rows: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    output_dim: int | None = None
    control_dim: int | None = None
    target_dim: int | None = None
    min_full_turn = max(cfg.output_memory, cfg.input_memory)

    for _, group in trajectories.sort_values("turn").groupby("trajectory_id", sort=False):
        group = group.sort_values("turn")
        if group.empty:
            continue
        r = group[list(cfg.target_columns)].iloc[0].to_numpy(dtype=float)
        ys = {
            int(row.turn): np.asarray(
                [getattr(row, col) for col in cfg.output_columns],
                dtype=float,
            )
            for row in group.itertuples()
        }
        us = {turn: _control_from_y_r(y, r, cfg.control_mode) for turn, y in ys.items()}
        output_dim = len(next(iter(ys.values())))
        control_dim = 0 if cfg.input_memory == 0 else len(next(iter(us.values())))
        target_dim = len(r)
        rows = {int(row.turn): row for row in group.itertuples()}
        for turn in sorted(ys):
            if turn < min_full_turn or turn + 1 not in ys:
                continue
            needed = {
                *(turn - lag for lag in range(cfg.output_memory)),
                *(turn - lag for lag in range(cfg.input_memory)),
                *(turn + 1 - lag for lag in range(cfg.output_memory)),
                *(turn + 1 - lag for lag in range(cfg.input_memory)),
            }
            if not needed.issubset(ys):
                continue
            row = rows[turn]
            next_row = rows[turn + 1]
            states.append(_state_at_turn(ys, us, turn, cfg))
            next_states.append(_state_at_turn(ys, us, turn + 1, cfg))
            targets.append(r)
            meta_rows.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "topic": getattr(row, "topic", ""),
                    "topic_split": getattr(row, "topic_split", ""),
                    "target_effective_norm": float(r[0]) if len(r) == 1 else r.tolist(),
                    "current_turn": int(turn),
                    "next_turn": int(turn + 1),
                    "y_t": float(ys[turn][0]) if output_dim == 1 else ys[turn].tolist(),
                    "y_next": float(ys[turn + 1][0])
                    if output_dim == 1
                    else ys[turn + 1].tolist(),
                    "target_raw_count": _optional_float(
                        getattr(row, "target_raw_count", np.nan)
                    ),
                    "raw_next": _optional_float(
                        getattr(next_row, "measured_raw_count", np.nan)
                    ),
                }
            )
    if not states:
        raise ValueError("no valid augmented-state transitions; check memory lengths and turns")
    return AugmentedStateDataset(
        meta=pd.DataFrame(meta_rows),
        Z_t=np.vstack(states),
        R=np.vstack(targets),
        Z_next=np.vstack(next_states),
        output_dim=int(output_dim or 0),
        control_dim=int(control_dim or 0),
        target_dim=int(target_dim or 0),
        output_memory=cfg.output_memory,
        input_memory=cfg.input_memory,
        control_mode=cfg.control_mode,
    )


def build_augmented_state_sequences(
    trajectories: pd.DataFrame,
    config: AugmentedStateConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build full augmented-state sequences for optional multi-step losses."""

    cfg = config or AugmentedStateConfig()
    sequences: list[tuple[np.ndarray, np.ndarray]] = []
    min_full_turn = max(cfg.output_memory, cfg.input_memory)
    for _, group in trajectories.sort_values("turn").groupby("trajectory_id", sort=False):
        group = group.sort_values("turn")
        if group.empty:
            continue
        r = group[list(cfg.target_columns)].iloc[0].to_numpy(dtype=float)
        ys = {
            int(row.turn): np.asarray(
                [getattr(row, col) for col in cfg.output_columns],
                dtype=float,
            )
            for row in group.itertuples()
        }
        us = {turn: _control_from_y_r(y, r, cfg.control_mode) for turn, y in ys.items()}
        state_rows = []
        for turn in sorted(ys):
            if turn < min_full_turn:
                continue
            needed = {
                *(turn - lag for lag in range(cfg.output_memory)),
                *(turn - lag for lag in range(cfg.input_memory)),
            }
            if needed.issubset(ys):
                state_rows.append(_state_at_turn(ys, us, turn, cfg))
        if len(state_rows) >= 2:
            sequences.append((np.vstack(state_rows), r))
    return sequences


class AugmentedKoopmanModel:
    """Affine Model III Koopman baseline.

    The model is

        z_(t+1) = A_z z_t + B_z r + c_z,  y_t = C z_t,

    where `C` extracts the first `output_dim` elements of the augmented state.
    """

    name = "augmented_koopman"

    def __init__(self, output_dim: int = 1, alpha: float = 1e-6):
        self.output_dim = int(output_dim)
        self.alpha = float(alpha)
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.c: np.ndarray | None = None

    def fit(self, Z_t: np.ndarray, R: np.ndarray, Z_next: np.ndarray) -> "AugmentedKoopmanModel":
        Z_t = _as_2d(Z_t)
        R = _as_2d(R)
        Z_next = _as_2d(Z_next)
        X = np.column_stack([Z_t, R, np.ones(len(Z_t))])
        theta = _ridge_solve(X, Z_next, self.alpha)
        state_dim = Z_t.shape[1]
        target_dim = R.shape[1]
        self.A = theta[:state_dim, :].T
        self.B = theta[state_dim : state_dim + target_dim, :].T
        self.c = theta[-1, :]
        return self

    def _check_fit(self) -> None:
        if self.A is None or self.B is None or self.c is None:
            raise RuntimeError(f"{self.name} has not been fit")

    def predict_next_z(self, z_t: np.ndarray, r: np.ndarray) -> np.ndarray:
        self._check_fit()
        Z = _as_2d(z_t)
        R = _as_2d(r)
        if len(R) == 1 and len(Z) > 1:
            R = np.repeat(R, len(Z), axis=0)
        assert self.A is not None and self.B is not None and self.c is not None
        pred = Z @ self.A.T + R @ self.B.T + self.c
        return pred[0] if np.asarray(z_t).ndim == 1 else pred

    def predict_y(self, z_t: np.ndarray) -> np.ndarray:
        Z = _as_2d(z_t)
        y = Z[:, : self.output_dim]
        return y[0] if np.asarray(z_t).ndim == 1 else y

    def rollout(self, z0: np.ndarray, r: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        states = [np.asarray(z0, dtype=float)]
        current = states[0]
        for _ in range(horizon):
            current = np.asarray(self.predict_next_z(current, r), dtype=float)
            states.append(current)
        state_arr = np.vstack(states)
        return state_arr, self.predict_y(state_arr)

    def diagnostics(self, horizon: int) -> dict[str, Any]:
        self._check_fit()
        assert self.A is not None and self.B is not None and self.c is not None
        out = controllability_diagnostics(self.A, self.B, horizon)
        out.update(
            {
                "model": self.name,
                "A": self.A.tolist(),
                "B": self.B.tolist(),
                "c": self.c.tolist(),
                "state_dim": int(self.A.shape[0]),
                "target_dim": int(self.B.shape[1]),
                "output_dim": self.output_dim,
            }
        )
        return out


@dataclass(frozen=True)
class DeepAugmentedKoopmanConfig:
    latent_dim: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "tanh"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 200
    lambda_rec: float = 1.0
    lambda_pred: float = 1.0
    lambda_latent: float = 0.1
    lambda_multi: float = 0.0
    multi_step_horizon: int = 0
    training_mode: str = "joint"
    dynamics_alpha: float = 1e-4
    random_state: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.training_mode not in {"joint", "reconstruction_then_ridge"}:
            raise ValueError(f"unsupported training_mode: {self.training_mode}")


def _activation(name: str) -> type[nn.Module]:
    if nn is None:
        raise ImportError("DeepAugmentedKoopmanAutoencoder requires PyTorch")
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
    if nn is None:
        raise ImportError("DeepAugmentedKoopmanAutoencoder requires PyTorch")
    layers: list[nn.Module] = []
    current = in_dim
    act = _activation(activation)
    for _ in range(num_layers):
        layers.extend([nn.Linear(current, hidden_dim), act()])
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class DeepAugmentedKoopmanAutoencoder:
    """Deep Autoencoder Koopman Model III.

    This model learns

        xi_t = E_theta(z_t)
        xi_(t+1) = K xi_t + B r + c
        z_hat_t = D_phi(xi_t)
        y_hat_t = C z_hat_t.
    """

    name = "deep_augmented_koopman"

    def __init__(
        self,
        state_dim: int,
        target_dim: int,
        output_dim: int = 1,
        config: DeepAugmentedKoopmanConfig | None = None,
        name: str | None = None,
    ):
        if torch is None or nn is None:
            raise ImportError("DeepAugmentedKoopmanAutoencoder requires PyTorch")
        self.state_dim = int(state_dim)
        self.target_dim = int(target_dim)
        self.output_dim = int(output_dim)
        self.config = config or DeepAugmentedKoopmanConfig()
        self.name = name or self.name
        torch.manual_seed(self.config.random_state)
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.encoder = _make_mlp(
            self.state_dim,
            self.config.latent_dim,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.activation,
        ).to(self.device)
        self.decoder = _make_mlp(
            self.config.latent_dim,
            self.state_dim,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.activation,
        ).to(self.device)
        self.K = nn.Parameter(
            0.05 * torch.randn(self.config.latent_dim, self.config.latent_dim, device=self.device)
        )
        self.B = nn.Parameter(
            0.05 * torch.randn(self.config.latent_dim, self.target_dim, device=self.device)
        )
        self.c = nn.Parameter(torch.zeros(self.config.latent_dim, device=self.device))
        self.training_history_: list[dict[str, float]] = []

    def parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            self.K,
            self.B,
            self.c,
        ]

    def autoencoder_parameters(self):
        return [*self.encoder.parameters(), *self.decoder.parameters()]

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.array(values, dtype=np.float32, copy=True), device=self.device)

    def _latent_step_tensor(self, xi: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return xi @ self.K.T + r @ self.B.T + self.c

    @staticmethod
    def _checkpoint_epoch(path: pathlib.Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    @classmethod
    def _latest_complete_checkpoint(cls, checkpoint_dir: pathlib.Path) -> pathlib.Path | None:
        candidates = []
        for path in checkpoint_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            missing = [
                name
                for name in ("state.pt", "_COMPLETE")
                if not (path / name).is_file()
            ]
            if missing:
                print(
                    f"[resume] skipping incomplete checkpoint {path}; "
                    f"missing={','.join(missing)}"
                )
                continue
            candidates.append(path)
        return max(candidates, key=cls._checkpoint_epoch) if candidates else None

    def _save_training_checkpoint(
        self,
        checkpoint_dir: pathlib.Path,
        optimizer: torch.optim.Optimizer,
        loader_generator: torch.Generator,
        epoch_completed: int,
    ) -> pathlib.Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_dir = checkpoint_dir / f"checkpoint-{epoch_completed:06d}"
        if final_dir.is_dir() and (final_dir / "_COMPLETE").is_file():
            return final_dir
        if final_dir.exists():
            print(f"[checkpoint] replacing incomplete checkpoint target {final_dir}")
            if final_dir.is_dir():
                shutil.rmtree(final_dir)
            else:
                final_dir.unlink()
        temporary_dir = checkpoint_dir / f".{final_dir.name}.tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)
        payload = {
            "epoch_completed": int(epoch_completed),
            "state_dim": self.state_dim,
            "target_dim": self.target_dim,
            "output_dim": self.output_dim,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "K": self.K.detach().cpu(),
            "B": self.B.detach().cpu(),
            "c": self.c.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "loader_generator_state": loader_generator.get_state(),
            "training_history": self.training_history_,
            "training_mode": self.config.training_mode,
            "config": asdict(self.config),
        }
        torch.save(payload, temporary_dir / "state.pt")
        (temporary_dir / "_COMPLETE").write_text("complete\n")
        os.replace(temporary_dir, final_dir)
        return final_dir

    def _load_training_checkpoint(
        self,
        checkpoint_path: pathlib.Path,
        optimizer: torch.optim.Optimizer,
        loader_generator: torch.Generator,
    ) -> int:
        payload = torch.load(
            checkpoint_path / "state.pt",
            map_location=self.device,
            weights_only=False,
        )
        dimensions = (payload["state_dim"], payload["target_dim"], payload["output_dim"])
        expected = (self.state_dim, self.target_dim, self.output_dim)
        if dimensions != expected:
            raise ValueError(
                f"checkpoint dimensions {dimensions} do not match model dimensions {expected}"
            )
        checkpoint_mode = payload.get("training_mode", "joint")
        if checkpoint_mode != self.config.training_mode:
            raise ValueError(
                f"checkpoint training mode {checkpoint_mode!r} does not match "
                f"configured mode {self.config.training_mode!r}"
            )
        checkpoint_config = payload.get("config")
        if checkpoint_config is not None:
            current_config = asdict(self.config)
            # Extending the requested epoch count and moving a checkpoint between
            # CPU/GPU are valid resume operations. All optimization-relevant
            # settings must remain identical.
            ignored = {"num_epochs", "device"}
            mismatches = {
                key: (checkpoint_config.get(key), current_config.get(key))
                for key in current_config
                if key not in ignored
                and checkpoint_config.get(key) != current_config.get(key)
            }
            if mismatches:
                raise ValueError(
                    "checkpoint training configuration does not match the current "
                    f"run: {mismatches}"
                )
        self.encoder.load_state_dict(payload["encoder"])
        self.decoder.load_state_dict(payload["decoder"])
        with torch.no_grad():
            self.K.copy_(payload["K"].to(self.device))
            self.B.copy_(payload["B"].to(self.device))
            self.c.copy_(payload["c"].to(self.device))
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        loader_generator.set_state(payload["loader_generator_state"].cpu())
        self.training_history_ = list(payload.get("training_history", []))
        return int(payload["epoch_completed"])

    def _fit_latent_ridge(self, Z_t: np.ndarray, R: np.ndarray, Z_next: np.ndarray) -> None:
        z_t = self.encode(Z_t)
        z_next = self.encode(Z_next)
        R = _as_2d(R)
        X = np.column_stack([z_t, R, np.ones(len(R))])
        theta = _ridge_solve(X, z_next, self.config.dynamics_alpha)
        latent_dim = self.config.latent_dim
        target_dim = R.shape[1]
        with torch.no_grad():
            self.K.copy_(torch.as_tensor(theta[:latent_dim, :].T, device=self.device).float())
            self.B.copy_(
                torch.as_tensor(
                    theta[latent_dim : latent_dim + target_dim, :].T,
                    device=self.device,
                ).float()
            )
            self.c.copy_(torch.as_tensor(theta[-1, :], device=self.device).float())

    def fit(
        self,
        Z_t: np.ndarray,
        R: np.ndarray,
        Z_next: np.ndarray,
        multi_step_sequences: list[tuple[np.ndarray, np.ndarray]] | None = None,
        checkpoint_dir: str | pathlib.Path | None = None,
        checkpoint_every_epochs: int = 20,
        resume: bool = True,
    ) -> "DeepAugmentedKoopmanAutoencoder":
        Z_t = _as_2d(Z_t).astype("float32")
        R = _as_2d(R).astype("float32")
        Z_next = _as_2d(Z_next).astype("float32")
        dataset = TensorDataset(
            torch.from_numpy(Z_t),
            torch.from_numpy(R),
            torch.from_numpy(Z_next),
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.config.random_state)
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            generator=loader_generator,
        )
        optimized_parameters = (
            self.parameters()
            if self.config.training_mode == "joint"
            else self.autoencoder_parameters()
        )
        optimizer = torch.optim.AdamW(
            optimized_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        resolved_checkpoint_dir = pathlib.Path(checkpoint_dir) if checkpoint_dir else None
        if resolved_checkpoint_dir is not None and checkpoint_every_epochs < 1:
            raise ValueError("checkpoint_every_epochs must be >= 1 when checkpointing is enabled")
        start_epoch = 0
        if resolved_checkpoint_dir is not None and resume:
            latest = self._latest_complete_checkpoint(resolved_checkpoint_dir)
            if latest is not None:
                start_epoch = self._load_training_checkpoint(
                    latest,
                    optimizer,
                    loader_generator,
                )
                print(f"[resume] deep augmented Koopman from {latest} at epoch {start_epoch}")

        stop_requested = False

        def request_stop(_signum, _frame) -> None:
            nonlocal stop_requested
            stop_requested = True

        previous_handlers: dict[int, Any] = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, request_stop)

        mse = nn.MSELoss()
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                total = 0.0
                n_batches = 0
                for z_batch, r_batch, z_next_batch in loader:
                    z_batch = z_batch.to(self.device)
                    r_batch = r_batch.to(self.device)
                    z_next_batch = z_next_batch.to(self.device)
                    xi = self.encoder(z_batch)
                    z_rec = self.decoder(xi)
                    loss_rec = mse(z_rec, z_batch)
                    if self.config.training_mode == "joint":
                        xi_next_pred = self._latent_step_tensor(xi, r_batch)
                        z_next_pred = self.decoder(xi_next_pred)
                        xi_next_true = self.encoder(z_next_batch)
                        loss_pred = mse(z_next_pred, z_next_batch)
                        loss_latent = mse(xi_next_pred, xi_next_true)
                        loss = (
                            self.config.lambda_rec * loss_rec
                            + self.config.lambda_pred * loss_pred
                            + self.config.lambda_latent * loss_latent
                        )
                    else:
                        loss = self.config.lambda_rec * loss_rec
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach().cpu().item())
                    n_batches += 1
                if (
                    self.config.training_mode == "joint"
                    and self.config.lambda_multi > 0
                    and multi_step_sequences
                ):
                    optimizer.zero_grad()
                    multi_loss = self._multi_step_loss(multi_step_sequences, mse)
                    (self.config.lambda_multi * multi_loss).backward()
                    optimizer.step()
                    total += float(multi_loss.detach().cpu().item())
                    n_batches += 1
                epoch_completed = epoch + 1
                self.training_history_ = [
                    {"epoch": float(epoch_completed), "loss": total / max(n_batches, 1)}
                ]
                should_checkpoint = resolved_checkpoint_dir is not None and (
                    epoch_completed % checkpoint_every_epochs == 0
                    or epoch_completed == self.config.num_epochs
                    or stop_requested
                )
                if should_checkpoint:
                    saved = self._save_training_checkpoint(
                        resolved_checkpoint_dir,
                        optimizer,
                        loader_generator,
                        epoch_completed,
                    )
                    print(f"[checkpoint] {saved}")
                if stop_requested:
                    raise InterruptedError(
                        f"training interrupted after exact checkpoint at epoch {epoch_completed}"
                    )
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if self.config.training_mode == "reconstruction_then_ridge":
            self._fit_latent_ridge(Z_t, R, Z_next)
            self.training_history_ = [
                {
                    **(self.training_history_[-1] if self.training_history_ else {}),
                    "dynamics_alpha": float(self.config.dynamics_alpha),
                }
            ]
        return self

    def _multi_step_loss(
        self,
        sequences: list[tuple[np.ndarray, np.ndarray]],
        mse: nn.Module,
    ) -> torch.Tensor:
        losses = []
        horizon = max(1, int(self.config.multi_step_horizon))
        for z_seq_np, r_np in sequences:
            if len(z_seq_np) < 2:
                continue
            z_seq = self._tensor(z_seq_np)
            r = self._tensor(np.asarray(r_np, dtype=float).reshape(1, -1))
            xi = self.encoder(z_seq[:1])
            max_k = min(horizon, len(z_seq) - 1)
            for step in range(1, max_k + 1):
                xi = self._latent_step_tensor(xi, r)
                pred = self.decoder(xi)
                losses.append(mse(pred, z_seq[step : step + 1]))
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def encode(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xi = self.encoder(self._tensor(_as_2d(z)))
        out = xi.detach().cpu().numpy()
        return out[0] if np.asarray(z).ndim == 1 else out

    def decode(self, xi: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.decoder(self._tensor(_as_2d(xi)))
        out = z.detach().cpu().numpy()
        return out[0] if np.asarray(xi).ndim == 1 else out

    def predict_next_latent(self, xi: np.ndarray, r: np.ndarray) -> np.ndarray:
        XI = _as_2d(xi)
        R = _as_2d(r)
        if len(R) == 1 and len(XI) > 1:
            R = np.repeat(R, len(XI), axis=0)
        with torch.no_grad():
            out = self._latent_step_tensor(self._tensor(XI), self._tensor(R))
        arr = out.detach().cpu().numpy()
        return arr[0] if np.asarray(xi).ndim == 1 else arr

    def predict_next_z(self, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        xi = self.encode(z)
        xi_next = self.predict_next_latent(xi, r)
        return self.decode(xi_next)

    def predict_y(self, z: np.ndarray) -> np.ndarray:
        Z = _as_2d(z)
        y = Z[:, : self.output_dim]
        return y[0] if np.asarray(z).ndim == 1 else y

    def rollout(self, z0: np.ndarray, r: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        states = [np.asarray(z0, dtype=float)]
        xi = self.encode(states[0])
        for _ in range(horizon):
            xi = self.predict_next_latent(xi, r)
            states.append(np.asarray(self.decode(xi), dtype=float))
        state_arr = np.vstack(states)
        return state_arr, self.predict_y(state_arr)

    def diagnostics(self, horizon: int) -> dict[str, Any]:
        K = self.K.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        out = controllability_diagnostics(K, B, horizon)
        out.update(
            {
                "model": self.name,
                "K": K.tolist(),
                "B": B.tolist(),
                "c": self.c.detach().cpu().numpy().tolist(),
                "state_dim": self.state_dim,
                "latent_dim": self.config.latent_dim,
                "target_dim": self.target_dim,
                "output_dim": self.output_dim,
                "training_mode": self.config.training_mode,
                "dynamics_alpha": self.config.dynamics_alpha,
                "training_history": self.training_history_,
            }
        )
        return out


def one_step_predictions(model: Any, dataset: AugmentedStateDataset) -> pd.DataFrame:
    z_pred = model.predict_next_z(dataset.Z_t, dataset.R)
    z_pred = _as_2d(z_pred)
    y_pred = z_pred[:, : dataset.output_dim]
    y_true = dataset.Z_next[:, : dataset.output_dim]
    out = dataset.meta.copy()
    out["model"] = model.name
    out["stage"] = "one_step"
    out["y_true"] = y_true[:, 0] if dataset.output_dim == 1 else list(y_true)
    out["y_pred"] = y_pred[:, 0] if dataset.output_dim == 1 else list(y_pred)
    out["z_mse"] = np.mean((z_pred - dataset.Z_next) ** 2, axis=1)
    out["z_mae"] = np.mean(np.abs(z_pred - dataset.Z_next), axis=1)
    out["y_mse"] = np.mean((y_pred - y_true) ** 2, axis=1)
    out["y_mae"] = np.mean(np.abs(y_pred - y_true), axis=1)
    return out


def augmented_prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    return {
        "one_step_mse": float(predictions["y_mse"].mean()),
        "one_step_mae": float(predictions["y_mae"].mean()),
        "z_space_mse": float(predictions["z_mse"].mean()),
        "z_space_mae": float(predictions["z_mae"].mean()),
        "n": int(len(predictions)),
    }


def rollout_augmented_from_trajectories(
    model: Any,
    trajectories: pd.DataFrame,
    config: AugmentedStateConfig,
    observed_seed_turns: int | None = None,
) -> pd.DataFrame:
    """Roll out Model III from a full augmented state in each trajectory.

    ``observed_seed_turns`` permits a shared observed prefix across models with
    different memory orders. When omitted, rollout starts from the first state
    containing the complete configured history.
    """

    rows: list[dict[str, Any]] = []
    minimum_seed_turn = max(config.output_memory, config.input_memory)
    seed_turn = (
        minimum_seed_turn
        if observed_seed_turns is None
        else int(observed_seed_turns)
    )
    if seed_turn < minimum_seed_turn:
        raise ValueError(
            f"observed_seed_turns must be >= {minimum_seed_turn}; got {seed_turn}"
        )
    for _, group in trajectories.sort_values("turn").groupby("trajectory_id", sort=False):
        group = group.sort_values("turn")
        if group.empty or int(group["turn"].max()) <= seed_turn:
            continue
        r = group[list(config.target_columns)].iloc[0].to_numpy(dtype=float)
        ys = {
            int(row.turn): np.asarray(
                [getattr(row, col) for col in config.output_columns],
                dtype=float,
            )
            for row in group.itertuples()
        }
        us = {turn: _control_from_y_r(y, r, config.control_mode) for turn, y in ys.items()}
        rows_by_turn = {int(row.turn): row for row in group.itertuples()}
        z_current = _state_at_turn(ys, us, seed_turn, config)
        for turn in sorted(ys):
            row = rows_by_turn[turn]
            if turn < seed_turn:
                z_pred = np.full_like(z_current, np.nan)
                y_pred = ys[turn]
                uses_seed = True
            elif turn == seed_turn:
                z_pred = z_current
                y_pred = ys[turn]
                uses_seed = True
            else:
                z_current = np.asarray(model.predict_next_z(z_current, r), dtype=float)
                z_pred = z_current
                y_pred = model.predict_y(z_pred)
                uses_seed = False
            y_true = ys[turn]
            valid_z_true = turn >= seed_turn
            if valid_z_true:
                z_true = _state_at_turn(ys, us, turn, config)
                z_err = z_pred - z_true
                z_mse = float(np.mean(z_err**2))
                z_mae = float(np.mean(np.abs(z_err)))
            else:
                z_mse = np.nan
                z_mae = np.nan
            y_arr = np.asarray(y_pred, dtype=float).reshape(-1)
            y_err = y_arr - y_true
            rows.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "topic": getattr(row, "topic", ""),
                    "topic_split": getattr(row, "topic_split", ""),
                    "target_effective_norm": float(r[0]) if len(r) == 1 else r.tolist(),
                    "turn": turn,
                    "y_true": float(y_true[0]) if len(y_true) == 1 else y_true.tolist(),
                    "y_pred": float(y_arr[0]) if len(y_arr) == 1 else y_arr.tolist(),
                    "uses_observed_seed": uses_seed,
                    "observed_seed_turns": seed_turn,
                    "model": model.name,
                    "stage": "rollout_observed_seed",
                    "z_mse": z_mse,
                    "z_mae": z_mae,
                    "y_mse": float(np.mean(y_err**2)),
                    "y_mae": float(np.mean(np.abs(y_err))),
                }
            )
    return pd.DataFrame(rows)


def rollout_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    eval_rows = predictions[~predictions["uses_observed_seed"]].copy()
    if eval_rows.empty:
        return {
            "rollout_mse": np.nan,
            "rollout_mae": np.nan,
            "z_space_mse": np.nan,
            "z_space_mae": np.nan,
            "n": 0,
        }
    return {
        "rollout_mse": float(eval_rows["y_mse"].mean()),
        "rollout_mae": float(eval_rows["y_mae"].mean()),
        "z_space_mse": float(eval_rows["z_mse"].mean()),
        "z_space_mae": float(eval_rows["z_mae"].mean()),
        "n": int(len(eval_rows)),
    }
