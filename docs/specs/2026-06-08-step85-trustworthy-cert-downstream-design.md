# Step 85 — Structure → trustworthy certificate → downstream win (direction ③) — design spec

*Date: 2026-06-08 · Status: design, pre-implementation · Owner: SE3-EJEPA / Certified World Models · Compute: RTX 3080 (CUDA) WSL2, git-mediated; Phases 0–2 CPU-smoke-runnable*

> Refines direction ③ of `docs/specs/2026-06-08-three-directions-to-8-seed.md`. The seed left a hedge — *"inflated
> $\lambda_1$ → over-observes (wasteful) OR (depending on sign) mistimes → worse frontier"* (seed §③, metric bullet).
> This spec **resolves that hedge** (§2) and pins the build. Reuses two LANDED results (`step74`, `step79`); no new system.

## 1. Goal & success criteria

**Thesis (closes the paper's own loop).** *Structure buys a **trustworthy** certificate.* E2/`step83` already shows the
$\mathbb{Z}_N$-equivariant world model recovers the $N{=}40$ Lyapunov spectrum ($R^2\approx0.98$) while the dense MLP's
is garbage ($R^2<0$, $\lambda_1$ inflated $\sim3\!-\!4\times$: learned $\approx5.8\!-\!7.1$ vs true $\approx1.56$). This
step shows that faithfulness has a **downstream consequence**: an agent that schedules observation by the *equivariant*
model's certified horizon acts well; one that trusts the *non-equivariant* model's certified horizon acts badly — **not
sub-optimally, but because it trusts a wrong horizon.**

**Deliverable (one system: $N{=}40$ Lorenz-96, D2 active re-observation, no control):**

- **(A) Headline — certificate-isolated, fixed-budget frontier.** With the forecaster held FIXED, the re-observation
  schedule dictated by the *correct* certificate Pareto-dominates the one dictated by the *wrong* certificate, under a
  finite observation budget. The win is attributable **100% to the certificate** (only $\lambda_1$ differs).
- **(B) Diagnostic — calibration.** The conv certificate is calibrated (measured/certified horizon ratio $\approx1$ in
  the predictive regime); the MLP certificate is off by the $\lambda_1$-inflation factor ($\approx3$). This is the
  *mechanism* behind (A) and the INCONCLUSIVE gate.
- **(D) Companion — package end-to-end.** Agent E (conv WM + conv cert) vs Agent N (MLP WM + MLP cert), realistic; (A)
  explains away the forecaster/certificate confound that a reviewer would attack.

**Headline figure (one system, 3 seeds):** (a) calibration — measured vs certified horizon, conv on the diagonal, MLP
off by $\sim3\times$ (with bootstrap error bars, reuse `step78`); (b) the certificate-isolated budget frontier —
aggregate violation vs observation budget $B$, conv-cert schedule left-shifted $\sim3\times$ below MLP-cert schedule,
**same forecaster**; (c) the package companion — Agent E vs Agent N.

**PASS (anchor):** G0 ∧ G1 hold on $N{=}40$ Lorenz-96 under the honest gates of §6. **INCONCLUSIVE** (reported, never
loosened) if the conv certificate is not calibrated or the wrong certificate does not actually mistime.

## 2. The metric subtlety, resolved (the crux)

The naive "Agent E frontier beats Agent N frontier" carries **three stacked problems**; the design dissolves all three.

1. **Confound (forecaster vs certificate).** E and N differ in *two* ways — the forecaster ($\mathbb{Z}_N$-conv vs dense
   MLP) AND the certificate (correct vs inflated $\lambda_1$). A reviewer attributes any win to better *forecasting*, not
   a trustworthy *certificate*. **Fix:** hold the forecaster FIXED (use the faithful conv for the actual forecasting in
   the headline) and feed it *both* schedules — the only varying quantity is the $\lambda_1$ each certificate reports.
2. **Same-frontier collapse.** The step79 frontier (violation vs observation count) is a property of the *forecaster's*
   error growth (governed by the **true** $\lambda_1\approx1.56$), not of the certificate. If one freely sweeps the
   re-observation interval, both certificates trace the *same* curve — a wrong $\lambda_1$ merely **relabels** $\epsilon$.
   **Fix:** do **not** sweep the interval freely. Let each certificate *dictate* the interval ($I=T_1(\epsilon)$ in map
   steps), and sweep the **observation budget** instead. A wrong certificate then picks a *dominated* interval.
3. **Sign.** The MLP's $\lambda_1$ is *inflated*, so $T_1=\log(1/\epsilon)/\lambda_1$ is too **short** → it
   *over*-observes. Over-observation is *safe-but-wasteful* (the same "conservatism" complaint `step84` hit), not unsafe.
   **Fix:** under a **finite observation budget**, local over-observation becomes global **starvation** — the MLP-cert
   schedule front-loads reads $\sim3\times$ too densely, exhausts the budget, and leaves the episode tail **open-loop**,
   where the forecast diverges → genuinely *higher aggregate violation*. "Wasteful" converts into "worse."

**Resolved position (records the 拍定).** The headline (A) is the **certificate-isolated, fixed-budget** frontier —
forecaster fixed, interval dictated by each certificate, budget swept. (B) calibration is the mechanism/diagnostic and
the gate; (D) package is the realistic companion that (A) de-confounds. Formally, with episode length $L$, per-step true
horizon $H^\*(\epsilon)\approx \tfrac{1}{\lambda_1^{\text{true}}\,\Delta t_{\text{map}}}\log(1/\epsilon)$, a budget of $B$
re-observations covers $\approx B\cdot I$ steps; with $I_E\approx H^\*$ (conv, calibrated) and $I_N\approx H^\*/3$ (MLP,
inflated), the conv-cert schedule covers $\min(B\,H^\*,L)$ steps vs the MLP-cert's $\min(B\,H^\*/3,L)$ — $\sim3\times$ fewer — so the
MLP-cert open-loop tail, hence aggregate violation, is strictly larger for every $B<3L/H^\*$ (the MLP-cert schedule needs
$\sim3\times$ the budget to catch up). The dominance is the $\lambda_1$ ratio, nothing else.

## 3. System & models (reuse; do NOT re-derive)

**System.** $N{=}40$ Lorenz-96, $F=8$ (chaotic), the standard high-D chaos benchmark and the exact regime where E2's
structure/non-structure spectrum split lives (the MLP collapses at $N{=}40$ after tying conv through $N\le28$; `step83`).
D2 is **observation-only** — the autonomous ($u\equiv0$) $\Delta t$-map; no control, no planner.

**Forecasters (two, from `step74`).** The $\mathbb{Z}_N$-equivariant `L96CyclicConv` and the dense `L96MLP`, trained on
the SAME $N{=}40$ data with the multi-step rollout loss (`step74.train_model`). Their autonomous Lyapunov spectra (hence
$\lambda_1$) are already produced and cached at `papers/figures/step74_lorenz96_spectrum_seed{0,1,2}.json`
(`lambda_learned[0]` after the descending sort). Seeds $\{0,1,2\}$.

**Certificate & D2 machinery (from `step79`/`step78`).** Reuse, do **not** modify:
- `step79.certificate(model, mu, sd, N, eps, ...)` → $\lambda_1$ (+ bootstrap CI via `step78`) → $T_1(\epsilon)$.
- `step79.certified_T1_steps(cert)` → the certified horizon in **map steps** (the units fix: $T_1/\Delta t_{\text{map}}$,
  not $\mathrm{round}(T_1)$).
- `step79.empirical_forecast_horizon(model, mu, sd, N, eps, ...)` → the measured first-crossing forecast horizon (the
  load-bearing validation of whether the certificate predicts reality).
- `step78.bootstrap_spectrum_ci` / `horizon_interval` → error bars.

**Signature bridge (the only glue).** `step74`'s models are autonomous (`model(x)`); `step79`'s D2 functions expect an
action-conditioned `model(x, u)`. Wrap: `aw = lambda m: (lambda x, u: m(x))`. Pass `step74`'s normalization `(mu, sd)`.
(Lyapunov exponents are coordinate-invariant, so `step74`'s per-site `mu/sd` vs `step79`'s scalar frame is immaterial
to the certificate; D2 needs no planner-equivariance, so the scalar-frame requirement of `step79` Phase 3+ does not
apply here.)

## 4. Phases (staged so INCONCLUSIVE is detected early & cheap)

**Phase 0 — calibration & smoke gate (= B; resolves the two empirical unknowns).** For each model $m\in\{$conv,MLP$\}$,
seed $s$, at predictive $\epsilon$ (start $\epsilon=0.2$, the regime where `step79`/`step84` calibrate): read
$\lambda_1^m$, the certified horizon $H_{\text{cert}}^m=\texttt{certified\_T1\_steps}$, and the empirical horizon
$H_{\text{emp}}^m=\texttt{empirical\_forecast\_horizon}$. Report:
- **self-calibration** $\rho_{\text{self}}^m = H_{\text{emp}}^m/H_{\text{cert}}^m$ (does each model's own cert predict
  its own forecast horizon?), and
- **cert-isolated cross ratio** $\rho_{\times} = H_{\text{emp}}^{\text{conv}}/H_{\text{cert}}^{\text{MLP}}$ (does the MLP's
  *certificate* mistime a *faithful* forecaster's actual horizon? — expected $\approx \lambda_1^{\text{MLP}}/\lambda_1^{\text{conv}}\approx3$).

This phase is the cheap CPU-smoke (autonomous $N{=}40$ rollouts are cheap; the cost is one Benettin–QR per model, small /
cached). It is also **gate G0**.

**Phase 1 — certificate-isolated fixed-budget frontier (= A; headline).** Forecaster FIXED $=$ conv. Episode length
$L\approx10\!-\!20\times H_{\text{emp}}^{\text{conv}}$. Two schedules: interval $I_E=H_{\text{cert}}^{\text{conv}}$ vs
$I_N=H_{\text{cert}}^{\text{MLP}}\,(\approx I_E/3)$. Sweep observation budget $B$: re-observe (reset the conv forecast to
truth) every $I$ steps until $B$ observations are spent, then run open-loop to $L$; measure the **aggregate violation
rate** (fraction of the $L$ steps with relative forecast error $>\epsilon$). Plot aggregate violation vs $B$. **The
conv-cert schedule reaches low aggregate violation at $\sim1/3$ the budget the MLP-cert schedule needs** (equivalently,
at matched $B$ near the knee, conv-cert violation $\ll$ MLP-cert). Same forecaster throughout ⇒ the gap is the
$\lambda_1$ ratio. 3 seeds.

**Phase 2 — package companion (= D; realistic, reported not gated).** Agent E $=$ (conv forecaster + conv cert), Agent N
$=$ (MLP forecaster + MLP cert), each self-scheduled under the same budget sweep; aggregate-violation-vs-$B$ frontier.
This is the end-to-end story; Phase 1 certifies that any E-over-N gap is **not** merely forecaster quality.

## 5. Components & interfaces

- `experiments/step85_trustworthy_cert_downstream.py`
  - `autonomous(m)` — the `model(x,u)=m(x)` wrapper (the signature bridge of §3).
  - `load_or_train_models(N, seed)` — load `step74` conv/MLP @ $N{=}40$ (train via `step74.train_model` + cache if the
    figure JSON is absent); return `(conv, mlp, mu, sd)`.
  - `calibration(model, mu, sd, N, eps, seed)` — returns $\{\lambda_1, H_{\text{cert}}, H_{\text{emp}}, \rho_{\text{self}}\}$
    (wraps `step79.certificate` + `certified_T1_steps` + `empirical_forecast_horizon`). **Phase 0.**
  - `budget_frontier(forecaster, mu, sd, N, interval, B_list, L, eps, seed)` — for each budget $B$, run the re-observe-
    until-spent-then-open-loop rollout on the TRUE dynamics, return `[(B, aggregate_violation), ...]`. The one piece of
    genuinely new logic; ~a re-observation loop with a budget counter over `step74.attractor_traj` ground truth.
  - `run_phase1(...)` — conv forecaster, two intervals ($I_E, I_N$), the budget sweep → the cert-isolated frontier. **A.**
  - `run_phase2(...)` — conv/MLP self-scheduled package frontier. **D.**
  - `run(seed, device)` — Phase 0→1→2; honest gate prints; writes `papers/figures/step85_trustworthy_cert.{json,png}`.
- `tests/test_step85.py` — fast, no training:
  - **equivariance** (per project policy): `step74.L96CyclicConv` output commutes with `torch.roll` to float round-off,
    $f(\mathrm{roll}(x,s))=\mathrm{roll}(f(x),s)$ — and the MLP does **not** (a control showing the structural difference).
  - `certified_T1_steps` monotone decreasing in $\lambda_1$ and in $\epsilon$; matches $\log(1/\epsilon)/(\lambda_1\Delta t)$.
  - **inflation→shorter-horizon**: doubling a stub model's $\lambda_1$ halves $H_{\text{cert}}$ (the §2-problem-3 mechanism).
  - **budget→starvation**: on a deterministic stub, halving the interval at fixed $B$ halves the covered fraction, so the
    open-loop tail (hence aggregate violation) rises (the §2 resolution mechanism).
  - INCONCLUSIVE path: a constructed non-calibrating model yields an honest INCONCLUSIVE, not a loosened pass.

Reuse, do **not** modify: `step74` (N=40 conv/MLP + spectra + `attractor_traj`), `step78` (bootstrap CI +
`horizon_interval`), `step79` (`certificate`, `certified_T1_steps`, `empirical_forecast_horizon`).

## 6. Honest gates (never loosen — INCONCLUSIVE instead)

- **G0 (conv certificate is trustworthy — the load-bearing premise):** $\rho_{\text{self}}^{\text{conv}}$ in a calibrated
  band (measured/certified $\in[\tfrac12,2]$, i.e. $|\log\rho|\le\log2$) on $\ge2/3$ seeds in the predictive regime. The
  whole of ③ rests on the equivariant certificate being correct; if it is not, report **INCONCLUSIVE** — do not proceed.
- **G1 (headline — the wrong certificate mistimes, and it costs):** the conv-cert schedule's aggregate violation
  $<$ the MLP-cert schedule's **across the swept budget** (Pareto dominance) on $\ge2/3$ seeds — the headline number
  reported at the knee $B^\*\approx L/H^\*_{\text{conv}}$ (conv has just covered the episode while the MLP schedule is
  still starved) — **AND** the gap is attributable to the $\lambda_1$ inflation (Phase 0: $\rho_\times\approx\lambda_1^{\text{MLP}}/\lambda_1^{\text{conv}}$,
  the wrong cert genuinely $\sim3\times$ off). **INCONCLUSIVE** if the inflated $\lambda_1$ does not actually mistime
  re-observation (e.g. the run's MLP is not inflated, or the budget never binds).
- **G2 (package — reported, not gated):** Agent E vs Agent N end-to-end. Reported as the realistic companion; the gated
  scientific claim is G1, which de-confounds it.

## 7. Compute path

Phases 0–2 are **CPU-smoke-runnable** on a Mac (autonomous $N{=}40$ rollouts are cheap; the only nontrivial cost is the
Benettin–QR $\lambda_1$ estimate per model, which is small and cacheable from `step74`). Promote to the **RTX 3080** only
to run full seeds × budget grid × bootstrap for publication error bars (`docs/remote_3080.md`, git-mediated). The seed's
"`step85`. Mostly Mac-runnable (reuses cached spectra); 3080 optional" holds.

## 8. Where it lands

- **paper2:** a full section — *"the certificate's trustworthiness changes a decision, attributed to structure"* — the
  E2 spectrum-faithfulness result (`step83`) given a downstream consequence; closes the loop structure → faithful
  spectrum → trustworthy certificate → better-acting agent.
- **ICLR:** a short experiment paragraph + the headline figure; ties the existing decision line (`step5`/`step79`/`step84`)
  to E2, and answers "does the certificate ever *change* an action because it is *trustworthy*?" — within-method, so the
  $\sim2\times$ conservatism that made `step84`'s return-win INCONCLUSIVE **cancels** (it hits both schedules).
- **Relation to `step84`:** `step84` showed the certificate is *sound but $\sim2\times$ conservative* on return; ③ stops
  asking the certificate to be the *optimum* and asks for a win where its *proven* property (a faithful, calibrated
  horizon from structure) is exactly what is needed — and the within-method contrast makes the constant factor irrelevant.

## 9. Risks & honest probabilities

- **Phase 0 — conv calibrates at $N{=}40$ D2 (G0):** ~0.65. `step84` predictive-regime ratio $0.93$ and `step79`
  $\epsilon{=}0.2$ calibration support it; D2@$N{=}40$ specifically is unverified until the smoke runs.
- **Phase 1 — headline lands (G1):** ~0.6. Near-arithmetic once $\lambda_1^{\text{MLP}}$ is inflated and the budget binds;
  it is **immune** to MLP forecasting quality (forecaster is the conv). Residual risk: G0 fails, or the "fixed a-priori
  budget" framing reads as engineered (mitigation: sweep $B$, show Pareto dominance, not a cherry-picked point).
- **Phase 2 — package companion (G2):** ~0.5 — carries the confound/collapse risks by construction, which is *why* it is
  the companion and not the gated claim.
- **Overall — ③ lands a clean, defensible result that moves the paper toward 8:** ~0.6 (consistent with the seed's
  0.5–0.55 for ③; cert-isolation lifts it slightly by removing the main attack surface).

## 10. Out of scope (YAGNI)

- **C — heterogeneous-allocation across a forcing-$F$ sweep** → deferred to `step85b`
  (`docs/specs/2026-06-08-step85b-spectrum-allocation-seed.md`); it binds the *full* spectrum $R^2$ (not just $\lambda_1$),
  needs per-$F$ retraining and the 3080, and carries an un-de-risked "is the MLP's $\lambda_1(F)$ ranking actually
  scrambled, or merely inflated-but-monotone?" risk — run only after this step's G1 passes.
- **① safety/abstention bound (`step86`), ② cert-gated MBRL (`step87`)** — separate fresh-session seeds (the three-
  directions seed).
- **No new chaotic system** beyond $N{=}40$ Lorenz-96 (reuse `step74`); **no control / planning loop** (D2 is
  observation-only — actionable control is `step5`/`step79`'s concern); **no spectrum field beyond $\lambda_1$** here (the
  re-observation interval reads only the leading exponent; the full-spectrum story is `step85b`'s).

## 11. Revision 2026-06-09a — after G0 (Phase 0 ran: 3/3 PASS, with three refinements)

Phase 0 ran on $N{=}40$ Lorenz-96, $\epsilon=0.2$, 3 seeds, full training + Benettin-QR
(`experiments/step85_trustworthy_cert_downstream.py`, result `papers/figures/step85_phase0_calibration.json`).
**G0 PASSED** — the conv certificate is calibrated on 3/3 seeds (self-ratio $H_{\rm emp}/T_1^{\rm steps}=1.17,0.63,0.82$,
all in $[\tfrac12,2]$); the MLP $\lambda_1$ is inflated $3.19\times,3.50\times,3.46\times$ (matching the cached step74
spectra), so its certified horizon is $24\!-\!28$ steps. Direction ③ is alive — proceed to Phase 1. Three data-driven
refinements (they *strengthen* the thesis but **flip how the win must be framed** — load-bearing):

1. **The forecasters are empirically MATCHED — so the confound is absent for free.** Conv empirical horizon
   $62\!-\!104$ steps; MLP empirical horizon $64\!-\!73$ steps (seed 1 the MLP even forecasts *slightly longer*); both
   one-step relMSE $\sim10^{-5}$. The dense MLP **forecasts values as well as the conv** at $N{=}40$ — it fails ONLY on
   the certificate (the Jacobian/spectrum). So the package contrast (D) is **de-confounded by the data itself** (equal
   forecasters ⇒ any gap is the certificate); the cert-isolated Phase 1 (A) remains the *rigorous* control, now
   corroborated rather than load-bearing-alone. Memorable framing: *the non-equivariant model forecasts fine but
   **cannot certify its own competence** — its broken Jacobian makes it needlessly distrust its own good forecasts.*

2. **The MLP certificate is PESSIMISTIC, not optimistic (the sign is settled) — so the budget framing is NECESSARY,
   not merely nice.** Inflated $\lambda_1$ ⇒ certified horizon $\sim3\times$ too **short** ⇒ the MLP-cert agent
   *over*-observes (its self-ratio $\approx2.5$ means it re-observes when the forecast is actually good for $\sim3\times$
   longer). **Without a budget this looks "safe"** — the over-observing MLP schedule would have *lower* naive violation
   than the (slightly optimistic) conv schedule, i.e. the naive metric would *favour the wrong model*. The win exists
   ONLY under the fixed-budget/fixed-$\epsilon$ framing of §2/§4 (over-observation → starvation → open-loop tail). G0
   thus **empirically vindicates** the §2 resolution and proves a free $\epsilon$/interval sweep would be an anti-result.
   Phase 1 must hold $\epsilon$ fixed and sweep the **budget**, exactly as specified.

3. **The conv certificate leans slightly OPTIMISTIC (mean self-ratio $0.87$; seed 1 $0.63$).** At its point-$T_1$ the
   conv schedule has modest in-window violation (it under-observes a touch). Two consequences for Phase 1: (a) use the
   **conservative end of the certificate CI** ($T_1^{\rm lo}$ from $\lambda_1^{\rm hi}$) for the conv schedule — this is
   the *sound* side of the bound, not tuning; (b) report the **full budget sweep** (the conv schedule's mild in-window
   violation is a small constant; the MLP schedule's starvation is the dominant effect, so conv dominates across the
   sweep regardless). Be honest in the writeup that the conv cert is calibrated-to-$\sim1.5\times$, not exact.

**Added Phase-1 control (preempts the top red-team attack).** D2 *gives error feedback at each re-observation*, so a
certificate-FREE **adaptive** agent (AIMD on the interval: lengthen if the last window stayed under $\epsilon$, shorten
if it crossed) can *learn* the right cadence and would beat both fixed schedules given enough observations. Include it as
a baseline and **scope the certificate's win to the tight-budget / few-shot / one-shot regime** — where the adaptive
agent cannot afford the exploration and the a-priori certificate's correct-from-step-1 cadence is what pays. (The
certificate's *necessity*, as opposed to warm-start value, lives in the no-feedback safety setting — direction ①.)
This is the honest ceiling of ③: a clean **mechanism + efficiency-under-budget** result that makes E2's $R^2$
consequential, not a safety/necessity claim.
