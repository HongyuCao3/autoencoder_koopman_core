# Thrust 3 half page: designing the command from the linear structure (draft, 2026-10-25)

> Spec: [`../experiments/controllability_closed_form_evidence3.md`](../experiments/controllability_closed_form_evidence3.md). Placement decided by the offline check: **future work**.
> Notation follows the paper here ($K, B, c_t, b$); in the proposal, map to $(A, B_p, c^p, d_k)$ per the #2 revision plan.

**Constraint satisfaction is linear in the command.** Under the affine model $s_{t+1}=Ks_t+Bc_t+b$, holding a command $c$ over the horizon gives
$s_{t+h|t}=K^h s_t+\sum_{j<h}K^j(Bc+b)$, so every predicted reading $\hat y_{t+h}=Cs_{t+h|t}$ is affine in $c$.
A requirement "$y$ stays in $[y_{\min},y_{\max}]$ for $h\le H$" is therefore $2H$ linear inequalities in $c$; for a scalar command the feasible set is an
interval $[c_{\min}(t),c_{\max}(t)]$ in closed form, and a quadratic tracking cost has the minimizer $\mathrm{clip}(-\beta/2\alpha,\,c_{\min},c_{\max})$.
The controller no longer has to enumerate candidates to find a feasible, cost-optimal command.

**Safety tightens the interval by the prediction-error bound.** Shrinking $[y_{\min},y_{\max}]$ inward by the $h$-step error bound $\Delta_h$ before solving
gives a command that keeps the true reading in range whenever the bound holds; this is the box-constraint special case of the robust formulation with
$u$ replaced by $c^p$.

**One scalar command reaches a line segment, so multi-attribute control is rank-limited.** The $H$-step reachable set from $s_t$ lies on the segment through
$K^Hs_t$ along $\sum_{j<H}K^jB$. Controlling two or three attributes jointly (Joint-2/3) with one scalar command is structurally rank-deficient (rank $\le1$
against two or three targets), which is the control-theoretic form of "extend to multi-axis commands" and motivates adding internal channels, i.e.
new columns of $B$. Where the command is genuinely multi-valued and identified (CEFR simplification, three action columns), the controllability matrix
over $H=4$ has full rank 4, but its smallest singular value is 36x below the largest (Gramian condition number 1286): reachable in principle, expensive in one direction.

**Status.** On 2297 logged decisions of our CEFR controller, the closed-form command matches the enumerating MPC's choice 87.6% of the time (pre-set bar 95%);
every mismatch is a decision where the MPC chose to stop, which a single command held over the horizon cannot express. We therefore present closed-form
design as future work (Thrust 3), not as a preliminary result.
