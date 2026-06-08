# Step 5 — Certified-Control Co-Demonstration (design spec)

*Date: 2026-06-07 · Status: design, pre-implementation · Owner: SE3-EJEPA / Certified World Models*

## 1. Goal & success criteria

Close two structural gaps the reviewers/red-team flagged:

- **The two certificate axes never meet on one system.** The *configuration* axis (orbit-flatness) is shown on
  PushT / SO(3) / pixels; the *horizon* axis (certified $T_j(\epsilon)$) is shown on Lorenz / Hénon / Rössler /
  Lorenz-96. No single system carries both.
- **The certificate never *changes a decision*.** It is computed and validated, but never shown to make a planner
  act better.

**Deliverable.** On a chaotic system with an *exact* symmetry, demonstrate **simultaneously**:

- **(C) Configuration axis** — an equivariant world model *plus an equivariant planner* yields **orbit-flat control**:
  rotating (shifting) the initial condition rotates the control signal exactly (ratio $\to 1.000$).
- **(H) Horizon axis** — a certified per-channel horizon $T_j(\epsilon)$ read from the world model's spectrum, **with
  bootstrap error bars** (reuse `step78`).
- **(D) Decision** — a **certificate-aware** MPC planner (planning depth $H = T_j(\epsilon)$) achieves **lower control
  cost** than a **horizon-blind** planner (fixed, too-long $H$), because the blind planner trusts the model *past* its
  certified horizon, where the rollout is wrong, and is misled into bad control. *The certificate tells you, a priori,
  the correct planning horizon — that is what it earns.*

**Headline figure (one system):** (a) control-orbit-flatness ratio $\approx 1.000$; (b) $T_j \pm$ error bars;
(c) certificate-aware vs. horizon-blind control cost (former wins), 3 seeds.

**PASS (anchor):** all three hold on controlled Lorenz-96 under honest gates.

## 2. System: controlled Lorenz-96 (anchor) + corroborators (staged)

Per the brainstorm ("try each"), staged by feasibility — **not** three parallel from-scratch builds.

**Anchor (must land): controlled Lorenz-96.**
$$ \dot{x}_i = (x_{i+1}-x_{i-2})\,x_{i-1} - x_i + F + u_i,\qquad F=8\ (\text{chaotic}),\quad |u_i|\le u_{\max}. $$
Exact $\mathbb{Z}_N$ cyclic (shift) symmetry. **Reuses `step74`** dynamics + $\mathbb{Z}_N$-conv world model + spectrum
/ horizon / `step78` CI. Control task: **suppress chaos / stabilize to the unstable uniform fixed point** $x_i = F$ (the
$\mathbb{Z}_N$-symmetric chaos-suppression target — at $x_i{=}F$ the field vanishes and it is the well-known unstable
equilibrium the chaos surrounds), scored by time-averaged $\lVert x - F\mathbf{1} \rVert$.

**Corroborators (stretch; park like the ESN if they don't land cleanly):** (i) $\mathbb{Z}_N$ coupled-pendulum ring
(embodied narrative); (ii) chaotic double pendulum (familiar benchmark, $\mathbb{Z}_2$ contrast). Same 3-piece demo.
If $\ge 2$ systems land: **the certificate-driven control law holds across a *class* of chaotic symmetric systems** —
strictly stronger than one system, and mirrors the paper's existing class-lifts (Lorenz→Hénon→Rössler; SO(2)→SO(3)).

## 3. Components (units & interfaces)

1. **Controlled dynamics** `l96_controlled(x, u)` (reuse `step74.l96_rhs` + control term; RK4 step). *Interface:*
   $(x,u)\mapsto x'$. *Invariant:* $\mathbb{Z}_N$-equivariant — `l96(shift·x, shift·u) = shift·l96(x,u)`.
2. **World model** (action-conditioned): reuse `step74`'s $\mathbb{Z}_N$-conv (equivariant) + dense MLP (baseline),
   extended to predict $x_{t+1}$ from $(x_t,u_t)$; trained with the existing multi-step rollout loss.
3. **Certificate**: $T_j(\epsilon)$ from the WM Jacobian spectrum (reuse `step74` QR) $+$ `step78` bootstrap CI.
4. **Planner** — CEM-MPC over control sequences $u_{0:H}$, evaluating cost via WM rollout. *Equivariant* because the
   WM is equivariant and the cost is $\mathbb{Z}_N$-invariant (the paper's (A5) clause), so the optimal control
   transforms equivariantly. Two variants: **certificate-aware** $H=T_j(\epsilon)$ (binding/fastest channel);
   **horizon-blind** $H=$ a fixed horizon chosen *without* the certificate. To avoid strawmanning it we **sweep** the
   fixed $H$ and report the blind planner at its *best* fixed choice; the claim is that the certificate-aware planner
   ($H=T_j$, **no tuning**) matches or beats the best-swept blind $H$ — the certificate hands you the right horizon for free.
5. **Closed-loop runner** — roll the *true* controlled dynamics under the planner's first action (MPC), measure
   control cost and orbit-flatness (run from $x_0$ and $\text{shift}\cdot x_0$; check `control(shift·x0)=shift·control(x0)`).

## 4. Data flow

true dynamics → collect $(x,u,x')$ → train WM (equiv + baseline) → read certificate $T_j\pm$CI → MPC closed loop
(cert-aware vs. blind) on the *true* dynamics → metrics (cost, orbit-flatness ratio).

## 5. Honest gates

- **(C)** control-orbit-flatness ratio $<1.05$ for the equivariant planner (vs. baseline drift); else `INCONCLUSIVE`.
- **(H)** $T_j$ recovered (reuse `step74`/`step78` gates).
- **(D)** certificate-aware (untuned, $H{=}T_j$) control cost $\le$ the *best-swept* horizon-blind cost on $3/3$ seeds,
  **and** strictly $<$ a naive long-$H$ blind planner; report the full cost-vs-$H$ curve; `INCONCLUSIVE` if not consistent.

## 6. Testing

- `l96_controlled` is $\mathbb{Z}_N$-equivariant (commutes with shift), for random $(x,u)$.
- **Planner equivariance**: CEM control under a shifted IC equals the shifted control (adapt the existing
  `tests/test_step73_planner_equivariance.py` pattern).
- Reuse `test_step74` (spectrum) and `test_step78` (CI).

## 7. Reproducibility

CPU/MPS; float64 for the spectrum; seeds 0/1/2; env knobs; honest gates. Heavy runs may go to the CUDA box.

## 8. Risks & mitigations

- **"Cert-aware beats blind" may be weak if the WM is too accurate** (then long-$H$ also works). *Mitigation:* pick a
  regime where the certified horizon is genuinely short relative to the task (strong chaos / coarse $\epsilon$), so a
  blind long-$H$ planner is genuinely misled. Report `INCONCLUSIVE` honestly if the effect does not appear.
- **CEM-MPC on L96 can be finicky.** Start with short horizons + moderate $u_{\max}$; validate the planner stabilizes
  at all before the cert-aware-vs-blind contrast.
- **Corroborators may not train/control cleanly.** Park them (no forcing); the anchor alone closes the gap.

## 9. Out of scope (YAGNI)

- No real robot / no pixels (the embodied configuration axis is already carried by PushT / SO(3) / pixels).
- No new theory — reuse Theorem A, the (A5) equivariant-planner clause, Theorem B, and the `step78` CI.
- No tuning of the blind baseline to *lose*; it gets a fair, commonly-chosen fixed horizon.
