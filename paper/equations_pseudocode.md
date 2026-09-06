# Step 3 (front half) — equations extracted as pseudocode from code

Extraction pass only (Sonnet). Variable names follow the code, not a
textbook Koopman formulation. A later Opus pass turns this into
`paper/equations.tex` and writes final `equation_operation` strings.

---

## 1. State construction (Markov / Memory-L / Augmented-L)

**Code location**: `src/koopman_ae/core.py`, `AugmentedStateConfig` (lines 77-128),
`_state_at_turn` (169-177), `build_augmented_state_dataset` (180-271).
Variant selection itself lives outside this file, in
`scripts/train.py::_state_config` (lines 98-141) — confirmed by direct read.

```pseudocode
# ---- variant -> (output_memory, input_memory), scripts/train.py::_state_config ----
if family == "markov":
    output_memory, input_memory = 1, 0                  # lag ignored
elif family == "memory":
    output_memory, input_memory = lag + 1, 0
else:  # family == "augmented"
    output_memory, input_memory = lag + 1, lag + 1

# ---- per trajectory group (rows sorted by turn) ----
r         = target_columns row-0 value                       # shape (target_dim,)
ys[turn]  = output_columns value at that turn                # shape (output_dim,)

_control_from_y_r(y, r, control_mode):
    r_aligned = broadcast/truncate r to len(y)   # scalar r repeated, or r[:len(y)] if longer
    error = r_aligned - y
    if control_mode == "error":
        return error                                          # shape (output_dim,)
    else:  # "error_abs_sign"
        return concat([error, abs(error), sign(error)])       # shape (3*output_dim,)

us[turn] = _control_from_y_r(ys[turn], r, control_mode)

# ---- state at a turn: concatenation of lagged output+control terms ----
_state_at_turn(ys, us, turn, config):
    y_parts = [ys[turn], ys[turn-1], ..., ys[turn-(output_memory-1)]]
    u_parts = [us[turn], us[turn-1], ..., us[turn-(input_memory-1)]]
    z_t = concat(y_parts + u_parts)   # shape (output_memory*output_dim + input_memory*control_dim,)

# ---- dataset assembly: one row per usable turn t per trajectory ----
for t in sorted(ys) where t >= max(output_memory, input_memory) and (t+1) in ys
          and all needed lagged turns exist in ys:
    Z_t[i]    = _state_at_turn(ys, us, t,   config)
    Z_next[i] = _state_at_turn(ys, us, t+1, config)
    R[i]      = r
# stack rows across all trajectories -> Z_t, R, Z_next matrices (state_dim = Z_t.shape[1])
```

**operation_name candidate**: concatenate a fixed number of lagged output
values (and, for the augmented family, an equal number of lagged control
values derived as the tracking error `r - y`, optionally extended with
`abs`/`sign`) into one delay-embedded state vector `z_t`; the term count is
set externally per variant (`markov`=1 term, `memory-L`/`augmented-L` =
`L+1` terms).

**Discrepancy note**: none noted. The module docstring's claims (e.g.
`output_memory=4` reaches 3 turns back, `lag=L -> L+1` terms, `markov`
ignores `lag`) were checked directly against `scripts/train.py::_state_config`
and match exactly.

---

## 2. AE lift + latent linear dynamics `ξ_{t+1} = Kξ_t + Br + c`

**Code location**: `src/koopman_ae/core.py`, `DeepAugmentedKoopmanAutoencoder`
class: `__init__` (445-506), `_latent_step_tensor` (514-515), `encode`/`decode`
(999-1009), `predict_next_latent` (1011-1019), `predict_next_z` (1021-1024).

```pseudocode
# parameters
encoder: MLP(state_dim -> latent_dim), decoder: MLP(latent_dim -> state_dim)
K: (latent_dim, latent_dim) parameter
B: (latent_dim, target_dim) parameter
c: (latent_dim,) parameter

# lift
xi = encoder(z)                      # z: (batch, state_dim) -> xi: (batch, latent_dim)

# one latent linear-dynamics step (note: right-multiplication by transposes)
_latent_step_tensor(xi, r):
    return xi @ K.T + r @ B.T + c    # xi:(batch,latent_dim), r:(batch,target_dim) -> (batch,latent_dim)

# un-lift
z_hat = decoder(xi)                  # xi:(batch,latent_dim) -> z_hat:(batch,state_dim)

# readout (not learned; a fixed slice)
y_hat = z_hat[:, :output_dim]

# rollout (predict_next_z / rollout()):
xi_0 = encoder(z_0)
for step in range(horizon):
    xi_{t+1} = _latent_step_tensor(xi_t, r)     # r held fixed across the rollout
    z_hat_{t+1} = decoder(xi_{t+1})
```

**operation_name candidate**: encode the (delay-embedded) state into a
latent vector, advance the latent vector one step by an affine map
`xi @ K.T + r @ B.T + c` driven by the (fixed, per-trajectory) target/control
vector `r`, then decode back to state space and read the output off a fixed
slice of the decoded state.

**Discrepancy note**: the class docstring writes `xi_(t+1) = K xi_t + B r + c`
(left-multiplication, standard notation), but the executed line is
`xi @ self.K.T + r @ self.B.T + c` (batched row-vector convention). These are
mathematically the same map once transposes are accounted for, but the
docstring's literal symbol order is not what is executed — flag this so the
LaTeX pass writes the matrix/vector shapes consistent with row-vector-batch
convention if it wants to match the code's actual tensor layout.

---

## 3. Two-stage `reconstruction_then_ridge` closed-form solve

**Code location**: `src/koopman_ae/core.py`, `_ridge_solve` (39-42),
`_fit_latent_ridge` (712-728); the two-stage control flow lives in
`fit()` (see the `if self.config.training_mode == "reconstruction_then_ridge"`
branch, lines 967-974, and the branch in the per-batch loop, 859-860, where
only `loss_rec` is descended by gradient in this mode).

```pseudocode
# Stage 1 (gradient descent, per-batch loop, training_mode == "reconstruction_then_ridge"):
for each batch (z_batch, r_batch, z_next_batch):
    xi     = encoder(z_batch)
    z_rec  = decoder(xi)
    loss   = lambda_rec * MSE(z_rec, z_batch)     # ONLY this term has gradients in this mode
    loss.backward(); optimizer.step()             # updates encoder/decoder only (autoencoder_parameters())
# K, B, c are NOT touched by this loop in this mode.

# Stage 2 (closed-form ridge, run once after Stage-1 training finishes, _fit_latent_ridge):
z_t    = encode(Z_t)          # no-grad forward pass through the now-fixed encoder, shape (n, latent_dim)
z_next = encode(Z_next)       # shape (n, latent_dim)
X      = column_stack([z_t, R, ones(n)])     # shape (n, latent_dim + target_dim + 1); appended intercept column

_ridge_solve(X, Y=z_next, alpha=dynamics_alpha):
    reg = alpha * I(X.shape[1])
    reg[-1, -1] = 0.0                        # the intercept column's own ridge penalty is zeroed out
    theta = pinv(X.T @ X + reg) @ X.T @ Y    # theta: (latent_dim+target_dim+1, latent_dim)
    return theta

theta = _ridge_solve(X, z_next, dynamics_alpha)
K = theta[:latent_dim, :].T               # (latent_dim, latent_dim)
B = theta[latent_dim:latent_dim+target_dim, :].T   # (latent_dim, target_dim)
c = theta[-1, :]                          # (latent_dim,), the unregularized intercept row
# K, B, c copied into the model's parameters (torch.no_grad())
```

**operation_name candidate**: fit the encoder/decoder by gradient descent on
reconstruction loss alone, then, holding the encoder fixed, solve a single
closed-form ridge-regularized least-squares problem for the latent dynamics
matrices `(K, B, c)` from an augmented design matrix `[xi_t, r, 1]`, with the
appended constant-1 column's own ridge coefficient forced to zero so the
intercept is unregularized while `K`/`B` are ridge-penalized.

**Discrepancy note**: none noted — `_fit_latent_ridge` and `_ridge_solve` do
exactly what the two-stage description implies; the only subtlety (the
zeroed-out `reg[-1,-1]`) is not mentioned in any docstring/comment near
`_ridge_solve` at all, so it is worth stating explicitly in the LaTeX rather
than assuming a reader would notice it.

---

## 4. Joint loss's four additive terms

**Code location**: `src/koopman_ae/core.py`, per-batch loss in `fit()`
(lines 845-864, 866-892) and the equivalent no-grad restatement in
`_eval_loss` (517-576).

```pseudocode
# per training batch (z_batch, r_batch, z_next_batch), training_mode == "joint":
xi            = encoder(z_batch)
z_rec         = decoder(xi)
loss_rec      = MSE(z_rec, z_batch)                       # term 1: reconstruction

xi_next_pred  = _latent_step_tensor(xi, r_batch)          # = xi @ K.T + r_batch @ B.T + c
z_next_pred   = decoder(xi_next_pred)
xi_next_true  = encoder(z_next_batch)
loss_pred     = MSE(z_next_pred, z_next_batch)            # term 2: one-step prediction in STATE space
loss_latent   = MSE(xi_next_pred, xi_next_true)           # term 3: one-step prediction in LATENT space

loss = lambda_rec * loss_rec + lambda_pred * loss_pred + lambda_latent * loss_latent
loss.backward(); optimizer.step()      # ONE optimizer step per batch, over ALL parameters (encoder+decoder+K+B+c)

# term 4 (conditional, once per EPOCH not per batch), only if lambda_multi > 0 and multi_step_sequences given:
_multi_step_loss(sequences, mse):
    for (z_seq, r) in sequences:
        xi = encoder(z_seq[0:1])
        for step in 1..min(multi_step_horizon, len(z_seq)-1):
            xi   = _latent_step_tensor(xi, r)              # repeated latent rollout, r fixed
            pred = decoder(xi)
            losses.append(MSE(pred, z_seq[step:step+1]))
    return mean(losses)                                     # term 4: multi-step rollout loss

weighted_multi_loss = lambda_multi * _multi_step_loss(...)
weighted_multi_loss.backward(); optimizer.step()   # SEPARATE optimizer step, once per epoch (not folded into the batch loss above)
```

**operation_name candidate**: jointly weight four MSE terms — state-space
reconstruction, one-step state-space prediction, one-step latent-space
prediction, and (when enabled) a multi-step latent-rollout term — but
descend them via two different optimizer-step schedules: the first three
terms get one gradient step per minibatch, while the fourth term gets one
additional gradient step per epoch, so no single scalar loss value is
literally descended by a single step; the reported "loss" is the weighted
sum only for logging/early-stopping purposes.

**Discrepancy note**: this is documented candidly in the code's own comment
block above `_eval_loss` (lines 525-553) and reproduced faithfully in
`_eval_loss` and `fit()`'s alternating-step structure — no gap between
comment and executed code was found here. Flagging only because it is a
genuine departure from "the joint loss is a single 4-term sum descended
jointly," which a naive equation write-up would otherwise imply; the LaTeX
pass should state the two-schedule structure explicitly rather than writing
one static weighted-sum equation as if it were literally what one
`.backward()` call descends.

---

## 5. Defense-task `z_t/v_t/y_t` schema and its dynamics with interaction terms

**Code location**: `persona_drift_control/src/persona_drift/modeling/dataset.py`,
`ReducedStateConfig` (32-107), `build_reduced_state_pairs` (163-202);
`persona_drift_control/src/persona_drift/modeling/koopman.py`,
`KoopmanSurrogate._psi`/`fit`/`step`/`readout` (105-149);
`persona_drift_control/src/persona_drift/modeling/interaction_lift.py`,
`augment_with_interaction` (39-47), `InteractionLiftedSurrogate.step` (62-65).

```pseudocode
# ---- state schema (build_reduced_state_pairs), config: nu>=1, mu>=0, contemporaneous_v ----
shift = 1 if contemporaneous_v else 0
for t in range(start, len(traj_rows)-1):     # start = max(nu-1, mu-shift)
    tv = t + shift
    y_hist      = ys[t-nu+1 : t+1]           # nu terms, INCLUDES current y_t
    v_hist      = vs[tv-mu : tv]             # mu terms, EXCLUDES current-slot v
    y_hist_next = ys[t-nu+2 : t+2]
    v_hist_next = vs[tv-mu+1 : tv+1]
    z      = concat(y_hist, v_hist, aux_now)          # shape (nu+mu+len(aux_cols),)
    v      = [vs[tv]]                                  # the free/current control scalar, shape (1,)
    y      = ys[t]                                     # scalar readout target for this pair
    z_next = concat(y_hist_next, v_hist_next, aux_next)
    # rows containing any NaN in the constructed values are dropped

# ---- lifting + linear dynamics (KoopmanSurrogate) ----
_psi(z) = concat([z, extra_features_fn(z)])            # eta_t, shape (state_dim + n_extra,); ARX: extra_features_fn = zeros(0)

# fit(): stack Psi = [_psi(z) for z in Z], Psi_next = [_psi(z) for z in Z_next]
X     = hstack([Psi, V, ones(n,1)])                     # (n, d_psi + d_v + 1)
theta = solve(X.T @ X + ridge*I, X.T @ Psi_next)         # ridge normal-equations solve (not pinv)
A = theta[:d_psi].T                                      # (d_psi, d_psi)
B = theta[d_psi:d_psi+d_v].T                             # (d_psi, d_v)
b = theta[d_psi+d_v:].reshape(-1)                        # (d_psi,)
C = solve(Psi.T @ Psi + ridge*I, Psi.T @ Y).reshape(1,-1)  # separate ridge solve, readout only

# step / readout:
step(z, v):   eta = _psi(z);  eta_next = A @ eta + B @ v + b;  return eta_next[:state_dim]
readout(z):   return (C @ _psi(z)).item()

# ---- interaction-lifted variant (interaction_lift.py) ----
augment_with_interaction(V, Z, state_index=0):
    interaction = V[:, :1] * Z[:, state_index:state_index+1]   # v_t * z_t[state_index]  (state_index=0 -> current y)
    V_aug = hstack([V, interaction])                            # (n, 2): [v, v*y]
# fed into KoopmanSurrogate.fit as its V, so B becomes (d_psi, 2) = [B1, B2]

InteractionLiftedSurrogate.step(z, v):
    v_raw = float(v[0])
    v_aug = [v_raw, v_raw * z[state_index]]
    return surrogate.step(z, v_aug)     # i.e. eta_next = A@eta + B1*v + B2*(v*z[state_index]) + b
```

**operation_name candidate**: build a delay-embedded reduced state
`z_t = [y-history, v-history(, aux)]` and a scalar control `v_t`, lift the
state through an optional nonlinear dictionary `_psi`, fit an affine
one-step map `eta_{t+1} = A eta_t + B v_t + b` (plus a separate ridge fit
for the linear readout `y_t = C eta_t`) by ridge least squares; the
interaction variant instead fits a two-column control `[v_t, v_t*y_t]` so
the action's effect on the next state depends on the current state.

**Discrepancy note**: `koopman.py`'s module docstring calls the readout
model `y_t ~= C eta_t` "eq. 16" and the transition "eq. 15" from an external
PDF, and states "ARX ... is deliberately not a separate model class" — both
match the executed code (same `fit`/`step` path, `extra_features_fn` is the
only switch). No discrepancy found between docstring and code here. One
note worth carrying forward: `fit()` solves `A/B/b` via `solve(gram, ...)`
(ordinary ridge normal equations) rather than `pinv` as `_ridge_solve` in
`koopman_ae/core.py` does — these are two different ridge implementations
in the two repos, not the same helper reused.

---

## 6. MPC objective function and budget constraint (Phase J, k=1 case)

**Code location**: `persona_drift_control/src/persona_drift/control.py`,
`KoopmanMPCController._current_state` (211-230), `_remaining_budget`
(232-243), `_planning_steps` (245-257), `_simulate` (259-268),
`next_u_remind` (270-283).

```pseudocode
# state read off recorded history (mirrors dataset.py's schema/shift convention)
_current_state(history):
    nu, mu = state_config.nu, state_config.mu
    shift  = 1 if state_config.contemporaneous_v else 0
    if len(history) < min_len: return None       # forces action=0 for early turns
    z = concat(y_hist, v_hist)                    # same construction as build_reduced_state_pairs

# ---- recursive value function over the binary action tree, budget-aware ----
_simulate(z, action, remaining_steps, remaining_budget):
    z_next = surrogate.step(z, [action])
    value  = surrogate.readout(z_next) - (repeat_penalty if action else 0.0)
    if remaining_steps <= 0:
        return value
    budget_after = None if remaining_budget is None else remaining_budget - action
    candidates   = (0,1) if (budget_after is None or budget_after >= 1) else (0,)
    return value + max( _simulate(z_next, a, remaining_steps-1, budget_after) for a in candidates )

next_u_remind(turn, history):
    z = _current_state(history)
    if z is None: return 0
    remaining_budget = remind_budget - sum(prior u_remind in history)     # None if unbudgeted
    if remaining_budget is not None and remaining_budget <= 0: return 0
    steps = _planning_steps(turn)      # = min(horizon, episode_length - turn + 1), else just horizon
    best_action = argmax_{a in {0,1}} _simulate(z, a, steps-1, remaining_budget)
    return best_action

# ---- k=1 (remind_budget=1) specialization ----
# at the root decision: remaining_budget = 1 (assuming none spent yet)
#   action=1 -> budget_after = 0 -> every subsequent recursive call is FORCED to candidates=(0,)
#               i.e. the entire remaining tree collapses to the all-zeros branch
#   action=0 -> budget_after = 1 -> subsequent calls still choose freely over {0,1}
# so the k=1 objective compares:
#   value(spend now)  = readout(step(z,1)) - repeat_penalty + [rest of horizon forced to 0]
#   value(save it)     = readout(step(z,0)) + max over remaining horizon with budget still 1
```

**operation_name candidate**: brute-force enumerate the length-`horizon`
binary action tree, scoring each branch by the sum of predicted readouts
(readout of the surrogate's one-step-ahead state) minus a repeat penalty
for reminding, picking the first action of the best branch, where the
budget constraint prunes the tree at each node so that once the allotted
number of reminders (`remind_budget`, e.g. 1) is spent, every remaining
node's candidate action set is forced to `{0}`.

**Discrepancy note**: none noted against the class docstring, which
describes exactly this recursive/enumerative structure and the k=1 pruning
behavior. Worth flagging structurally for the LaTeX pass: this is NOT a
closed-form quadratic MPC objective — it is a discrete max-recursion over a
binary tree with a hard state-dependent (budget-dependent) constraint on
the admissible action set, so writing it as a continuous constrained
optimization (`min ... s.t. sum(u) <= budget`) would misrepresent what the
code executes; the actual constraint enforcement is the `candidates = (0,)`
branch pruning, not a Lagrangian or projection step.

---

## 7. Marginal-benefit expression `C(I+A)(B_1+B_2 y)` and closed-form threshold `y*`

**Code location**: NOT in code — derived/documented in
`docs/experiments/koopman_phaseI_policy_closed_form.md` (this doc's own
derivation, cross-checked by the doc's author against
`control.py::KoopmanMPCController._simulate` and a saved fitted-model JSON).
Per the task brief, extracted as stated rather than re-derived from code.

```pseudocode
# fitted matrices (nu=1, mu=2, contemporaneous_v=True), z = [y_{t-1}, u_{t-2}, u_{t-1}]:
# A (3x3), B1 = B[:,0], B2 = B[:,1], b (3,), C (1x3) -- values recorded in the doc

# 7.1 "textbook" one-step marginal-benefit expression (a documented SPECIAL CASE, not what the
#     real controller computes in general):
margin_1step(y) = C @ (I + A) @ (B1 + B2 * y)
y_star_linear: solve margin_1step(y) = 0
    C(I+A)      = [1.7036255, -0.0247579, -0.0548555]
    C(I+A)B1    = 0.4387906
    C(I+A)B2    = -0.5460389
    y_star_linear = - C(I+A)B1 / C(I+A)B2 = 0.803589

# 7.2 what KoopmanMPCController._simulate ACTUALLY computes at horizon=2 (the true margin):
step(z, a):  z' = A@z + B1*a + B2*(a*z[0]) + b;  readout = C @ z'
sim(z, a)  = C @ step(z, a) + max_{a2 in {0,1}} C @ step(step(z, a), a2)
margin_actual(z) = sim(z, 1) - sim(z, 0)
# margin_1step(y) == margin_actual ONLY when both branches' optimal second-step action is 0
#   (the doc calls this the special case; verified true on the observed y=1.0 grid point only)
y_star_actual: numeric root of margin_actual as function of y (holding z[2]=u_{t-1} fixed) = 0.788068

# 7.3 operational equivalence: y_probe lives on a {0, 0.25, 0.5, 0.75, 1.0} grid, and no grid
#     point falls between 0.788 and 0.804, so both thresholds imply the identical decision rule:
decision rule: remind iff y_{t-1} <= 0.75   (equivalently: iff y_{t-1} < y_star, for either y_star)
```

**operation_name candidate**: compute the closed-form one-step marginal
value of reminding under the linear+interaction surrogate as
`C(I+A)(B1 + B2*y)`, solve its zero-crossing for a threshold `y*`, and note
that this closed form is only an approximation to the actual two-step
MPC margin (which includes a second-step `max` over the controller's own
future action)—the two thresholds (0.8036 vs 0.7881) differ by 0.015 but
are operationally identical given the judge score's 0.25 grid spacing.

**Discrepancy note**: the doc itself is explicit that `C(I+A)(B_1+B_2 y)` is
NOT what the real controller computes — it is a textbook-style simplification
that only coincides with the actual controller's margin at one grid point
(`y=1.0`, where both continuations happen to choose the second-step action
0). The doc also warns against a related but wrong derivation: naively
iterating the fixed point `y <- C(I-A)^{-1}(b+B1+B2 y)` for the `u≡1`
steady state diverges (gain magnitude 1.081 > 1) because `v_aug=[1,y]`
is state-dependent, not a genuine constant input; the correct steady-state
computation folds the interaction term into the system matrix,
`A_tilde = A + B2 @ e1.T`, before inverting `(I - A_tilde)`. Any LaTeX
write-up of this equation must carry this caveat rather than presenting
`C(I+A)(B1+B2 y)` as the controller's literal decision rule.

---

## 8. Finite-horizon controllability matrix and Gramian

**Code location**: `src/koopman_ae/core.py`, `controllability_diagnostics`
(lines 45-74). (An independently-duplicated, field-name-identical copy also
exists at `persona_drift_control/src/persona_drift/modeling/koopman.py`
lines 50-92 — deliberately not imported/shared, per that file's own
docstring, to avoid a cross-package dependency.)

```pseudocode
controllability_diagnostics(A, B, horizon):
    gramian = zeros(A.shape[0], A.shape[0])
    Ak = I(A.shape[0])
    blocks = []
    for _ in range(horizon):
        block = Ak @ B                    # A^k @ B, shape (n, d_v)
        blocks.append(block)
        gramian += block @ block.T        # accumulate sum_k (A^k B)(A^k B)^T
        Ak = Ak @ A                        # advance to A^(k+1) for next iteration

    ctrb = concat(blocks, axis=1)          # (n, horizon*d_v): [B, AB, A^2B, ..., A^(horizon-1)B]
    singular_values = svd(ctrb, compute_uv=False)
    gramian_eigs    = eigvalsh(gramian)     # symmetric-matrix eigendecomposition
    eigvals         = eigvals(A)            # full (possibly complex) spectrum of A

    return {
        controllability_rank:        matrix_rank(ctrb),
        controllability_matrix:      ctrb,
        controllability_singular_values: singular_values,
        gramian:                     gramian,
        gramian_eigenvalues:         gramian_eigs,
        gramian_condition:           cond(gramian + 1e-12*I),
        A_eigenvalues_real/imag:     real/imag parts of eigvals,
        spectral_radius:             max(abs(eigvals)),
    }
```

**operation_name candidate**: build the finite-horizon controllability
matrix `[B, AB, A^2B, ..., A^(horizon-1)B]` by repeated left-multiplication
by `A`, take its rank and singular values, and separately accumulate the
finite-horizon controllability Gramian `sum_{k=0}^{horizon-1} (A^k B)(A^k B)^T`
and its eigenvalues/condition number, alongside the raw spectrum and
spectral radius of `A` alone.

**Discrepancy note**: none noted — the function has no surrounding
docstring beyond its one-line summary ("Finite-horizon controllability
diagnostics for linear controlled dynamics"), and the code matches that
description exactly. Two call sites feed it different `(A, B)` pairs:
`AugmentedKoopmanModel.diagnostics` and
`DeepAugmentedKoopmanAutoencoder.diagnostics` both pass their own fitted
system matrices — the function itself is agnostic to which Koopman variant
produced `A`/`B`.
