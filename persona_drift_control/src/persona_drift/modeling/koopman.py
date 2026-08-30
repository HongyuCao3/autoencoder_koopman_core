"""Controlled Koopman surrogate for persona-drift trajectories, following
Control_of_Foundational_Model_revised.pdf sections 4-7.

eta_t = [z_t, extra_features(z_t)]         (eq. 8, generalized to a lifting)
eta_(t+1) ~= A eta_t + B v_t + b           (eq. 15)
y_t       ~= C eta_t                       (eq. 16)

The reference matrix E is intentionally not modeled: DATA_COLLECTION_PROTOCOL.md
section 4 keeps `r` out of the prompt entirely during collection, so `r`
never varies in this data and E would be unidentifiable from it -- fitting
one anyway would silently produce an arbitrary, meaningless matrix rather
than a real reference-response direction.

ARX (linear autoregressive with exogenous input) is the special case
`extra_features_fn=no_extra_features` (eta_t == z_t, no lifting at all): it
is deliberately not a separate model class or a separate code path, so the
ARX baseline goes through the exact same fitting, ridge penalty, and
evaluation (`modeling.evaluate`) as any richer lifting. Any measured quality
difference between "the Koopman surrogate" and "the ARX baseline" is then
attributable only to the lifting dictionary, which is what makes the
comparison fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def no_extra_features(z: np.ndarray) -> np.ndarray:
    """ARX: the lifted dictionary is just z itself, no extra observables."""
    return np.zeros(0)


def abs_sign_extra_features(z: np.ndarray) -> np.ndarray:
    """A minimal genuinely-nonlinear lifting: abs and sign of the current
    y_probe reading. Mirrors koopman_ae.core.AugmentedStateConfig's
    `control_mode="error_abs_sign"` idea (asymmetric response to positive vs.
    negative deviations) -- applied here to the raw output history instead of
    a tracking error, since persona-drift's collection protocol has no
    reference `r` to build an error signal from. Assumes
    `ReducedStateConfig.nu == 1` (z[0] is then exactly the current y_t)."""

    current_y = z[0]
    return np.array([abs(current_y), np.sign(current_y)])


def controllability_diagnostics(A: np.ndarray, B: np.ndarray, horizon: int) -> dict:
    """Finite-horizon controllability rank/Gramian diagnostics
    (Control_of_Foundational_Model_revised.pdf eq. 23-26): C_T, its rank and
    singular values, the Gramian W_T and its eigenvalues/condition number,
    and A's eigenvalues/spectral radius.

    Field names intentionally match koopman_ae.core.controllability_diagnostics
    (same idea, same interface) but are NOT imported from there: that
    function is pure, dependency-free numpy with no coupling to Model III's
    schema, so duplicating ~15 lines here keeps `persona_drift_control`
    independently installable rather than introducing a cross-package
    dependency between two separately packaged sub-projects for something
    this small. Everything with real surface area -- data loading,
    splitting, fitting, evaluation -- is NOT duplicated like this; see
    dataset.py and evaluate.py for why Model III's own dataset builder does
    not apply here at all.
    """

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
        "gramian_condition": float(np.linalg.cond(gramian + 1e-12 * np.eye(A.shape[0]))),
        "A_eigenvalues_real": [float(v.real) for v in eigvals],
        "A_eigenvalues_imag": [float(v.imag) for v in eigvals],
        "spectral_radius": float(max(abs(v) for v in eigvals)) if len(eigvals) else float("nan"),
    }


@dataclass
class KoopmanSurrogate:
    extra_features_fn: Callable[[np.ndarray], np.ndarray] = no_extra_features
    ridge: float = 1e-6
    state_dim: int = field(default=0, init=False)
    A: np.ndarray = field(default=None, init=False)
    B: np.ndarray = field(default=None, init=False)
    b: np.ndarray = field(default=None, init=False)
    C: np.ndarray = field(default=None, init=False)

    def _psi(self, z: np.ndarray) -> np.ndarray:
        """eta = [z, extra_features(z)]. Keeping z as a verbatim prefix of
        eta (rather than lifting it away entirely) is what lets `step()`
        recover a next z_t (not just a next eta_t) from the linear update,
        so the surrogate can be rolled out for more than one step -- the
        "important qualification" in
        Control_of_Foundational_Model_revised.pdf section 4."""
        return np.concatenate([z, self.extra_features_fn(z)])

    def fit(self, dataset: dict[str, np.ndarray]) -> "KoopmanSurrogate":
        """dataset: output of modeling.dataset.build_identification_dataset.
        Solves eq. (18) (A, B, b) and eq. (19) (C) as ridge least squares."""

        Z, V, Z_next, Y = dataset["Z"], dataset["V"], dataset["Z_next"], dataset["Y"]
        if Z.shape[0] == 0:
            raise ValueError("empty identification dataset")
        self.state_dim = Z.shape[1]

        Psi = np.stack([self._psi(z) for z in Z])
        Psi_next = np.stack([self._psi(z) for z in Z_next])
        n, d_psi = Psi.shape
        d_v = V.shape[1]

        X = np.hstack([Psi, V, np.ones((n, 1))])
        gram = X.T @ X + self.ridge * np.eye(X.shape[1])
        theta = np.linalg.solve(gram, X.T @ Psi_next)
        self.A = theta[:d_psi].T
        self.B = theta[d_psi : d_psi + d_v].T
        self.b = theta[d_psi + d_v :].reshape(-1)

        gram_c = Psi.T @ Psi + self.ridge * np.eye(d_psi)
        self.C = np.linalg.solve(gram_c, Psi.T @ Y).reshape(1, -1)
        return self

    def step(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predict z_(t+1) from (z_t, v_t) -- the `Predictor.step` contract
        in modeling.evaluate."""
        eta = self._psi(z)
        eta_next = self.A @ eta + self.B @ np.asarray(v, dtype=float) + self.b
        return eta_next[: self.state_dim]

    def readout(self, z: np.ndarray) -> float:
        """Predict y_t from z_t -- the `Predictor.readout` contract in
        modeling.evaluate."""
        return float((self.C @ self._psi(z)).item())

    def controllability(self, horizon: int) -> dict:
        if self.A is None:
            raise RuntimeError("fit() must be called before controllability()")
        return controllability_diagnostics(self.A, self.B, horizon)
