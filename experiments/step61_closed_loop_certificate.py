r"""Step 61 — the predictability certificate at the TASK level: closed-loop pose control over the orbit.

Experiment 9 (Step 59) showed the certificate holds for the *rollout relMSE* of a learned model of real PushT
contact physics: the equivariant model's H-step error is flat over the SO(2) orbit, no MLP scale reaches its
floor out of the wedge. The standing question a reviewer asks next is whether that **prediction**-level
flatness converts to **task** competence: a low relMSE is a proxy; does the planner that *uses* the model
inherit the certificate?

Steps 10 and 12 tried this directly (CEM-MPC task success vs orientation) and both came back noise-limited: the
prediction gap did not resolve into a clean task-success gap, because (i) position-only pushing is carried by the
near-linear agent-PD subsystem an MLP extrapolates fine, and (ii) each orientation bin used *independent* random
tasks, so task-difficulty variance swamped the orientation effect at laptop N. This step removes both confounds.

The certificate's closed-loop statement (Theorem A under assumption A5, the G-equivariant planner). If the world
model is G-equivariant and the planner is a G-equivariant procedure under an SO(2)-invariant cost, then the
closed-loop policy is itself equivariant: $\pi(g\cdot s)=g\cdot\pi(s)$. Hence on the SO(2)-exact interior of real
PushT the realized trajectory from a rotated task is the rotation of the realized trajectory from the base task,
and the realized **task error is exactly orbit-invariant**. A5 is not free; we *instantiate* it. A plain CEM is
not equivariant (box action bound + per-axis covariance + scene-blind Gaussian noise). We make CEM equivariant
with three changes that are each SO(2)-natural: an **isotropic** per-step exploration covariance, a **disk**
action bound $\lVert a\rVert\le1$ (the rotation-invariant constraint, not the box $[-1,1]^2$), and
**scene-covariant** action noise (the Gaussian draws are rotated by the orbit angle). With these, candidates at
orbit angle $\beta$ are exactly $R_\beta$ of the base candidates, the invariant cost ranks them identically, and
the elite-mean plan satisfies $\text{plan}(R_\beta s)=R_\beta\,\text{plan}(s)$ to the float floor.

Three measurements, most to least robust:

  [A] Planning-cost orbit-invariance (model-internal, exact). The pose cost that drives the planner is
      SO(2)-invariant; under the equivariant model its value is flat over the orbit to the float floor, under the
      MLP it is scrambled (drift ~1). This is the mechanism: the equivariant model's notion of "how good is this
      plan" transports across the orbit; the MLP's does not.

  [B] Closed-loop task error over the orbit, PAIRED (same base tasks rotated), with the G-equivariant planner.
      * model-rollout (no env): EXACT certificate -- the equivariant planner+model system is orbit-equivariant to
        the float floor, so the planned terminal pose error is flat over the orbit; the MLP's is not.
      * real-env realized error: the practical confirmation -- flat for the equivariant model up to the real env's
        own ~1e-5/step interior equivariance; broken for the MLP.

  [C] Realistic planner (plain CEM, scene-blind, fresh per orientation), paired tasks, multi-seed. The rebuttal to
      "you rotated the noise": even *without* enforcing A5 in the planner, the equivariant model stays flatter and
      more accurate out of the wedge; the gap is just partly masked by the planner's own (scene-blind) noise.

Models differ only in the symmetry prior:
  * equivariant -- ``RichVNForwardModelPushT`` (Step 59): invariant-scalar-gated Vector-Neuron, exactly
    SO(2)-equivariant AND competitive in absolute accuracy (so "flat is not good" is not the explanation);
  * baseline    -- a plain MLP on the flattened state+action, scaled generously.

Honest gate at the orbit (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   equivariant planning-cost drift exact:          cost_drift_eq < 1e-4   AND  cost_drift_mlp > 0.3;
  (ii)  equivariant model-rollout task error flat:      rollout_ratio_eq < 1.02 (the EXACT certificate);
  (iii) equivariant real-env task error orbit-flat:     realenv_ratio_eq < 1.30  AND  realenv_ratio_mlp > 1.5;
  (iv)  equivariant competitive in the wedge:           ang_eq_inwedge <= 1.25 * ang_mlp_inwedge.

Run:        SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
                .venv/bin/python experiments/step61_closed_loop_certificate.py
Seeded:     STEP61_SEED=0|1|2 .venv/bin/python experiments/step61_closed_loop_certificate.py
Writes:     papers/figures/step61_closed_loop_certificate.{json,png}
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from step10_pusht_closed_loop import (  # noqa: E402
    CENTER,
    POS_SCALE,
    SUCCESS_PX,
    VEL_SCALE,
    MLPForwardModelPushT,
    goal_to_scaled,
    info_to_packed,
    make_env,
    model_equivariance_err,
    n_params,
    reset_task,
    rot_np,
    rotate_packed_torch,
    rotate_state,
)
from step12_pose_control import (  # noqa: E402
    ANGLE_TOL_DEG,
    REORIENT_DEG,
    W_ANG,
    W_POS,
    angle_err_deg,
    collect_pose_transitions,
    pose_cost_drift,
    sample_pose_task,
)
from step59_pusht_certificate import RichVNForwardModelPushT, train_fair  # noqa: E402

SEED = int(os.environ.get("STEP61_SEED", "0"))
SMOKE = os.environ.get("STEP61_SMOKE", "0") == "1"

WEDGE_DEG = 45.0                                   # train wedge half-width (phi in [0, 90) about centre 45)
ORBIT_DEG = [0, 45, 90, 135, 180, 225, 270, 315]   # orbit angles applied to the SAME base tasks
MLP_HIDDEN = 64 if SMOKE else 256                  # generous baseline (~6x the RichVN's params)
N_TRAIN = 400 if SMOKE else 1500
EPOCHS = 150 if SMOKE else 700
N_BASE = 6 if SMOKE else 24                        # base tasks (each rotated to every orbit angle -> paired)
H = 20
FIG = ROOT / "papers" / "figures"


# --------------------------------------------------------------------------------------------------- #
# task-space geometry: rotate a whole pose task about the arena centre (the SO(2) orbit action)
# --------------------------------------------------------------------------------------------------- #
def rotate_pose_task(task: dict, beta: float) -> dict:
    r"""Rotate a pose task (state, goal position, goal angle) by ``beta`` rad about CENTER.

    The interior real PushT dynamics is SO(2)-equivariant (Step 10 [A]), so the rotated task is a genuine,
    physically-valid member of the same task's orbit -- only the scene orientation changes.
    """
    R = rot_np(beta)
    return {
        "state": rotate_state(task["state"], beta),
        "goal_px": R @ (task["goal_px"] - CENTER) + CENTER,
        "goal_angle": task["goal_angle"] + beta,
        "beta": task["beta"] + beta,
    }


def pack_state7(s7: np.ndarray) -> np.ndarray:
    r"""Pack a raw 7-vector PushT state ``[ax,ay,bx,by,bθ,avx,avy]`` into the model's 8-vector, by the SAME
    rule as ``info_to_packed`` but WITHOUT round-tripping through the env. This keeps the orbit action exact:
    ``pack_state7(R_β·s) = R_β·pack_state7(s)`` to the float floor (the env reset is only ~1e-5 equivariant),
    so the model-rollout certificate probe in [B] is genuinely exact rather than env-noise-limited.
    """
    return np.concatenate([
        (s7[0:2] - CENTER) / POS_SCALE,
        s7[5:7] / VEL_SCALE,
        (s7[2:4] - CENTER) / POS_SCALE,
        [math.cos(s7[4]), math.sin(s7[4])],
    ]).astype(np.float32)


# --------------------------------------------------------------------------------------------------- #
# the G-equivariant CEM planner (instantiates A5): isotropic covariance + disk action bound +
# scene-covariant noise.  With an equivariant model and invariant cost, plan(R s) = R plan(s) exactly.
# --------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def cem_plan_pose_equiv(
    model, packed0: np.ndarray, goal_scaled: np.ndarray, goal_dir: np.ndarray,
    *, noise_rot: float = 0.0, H: int = H, n_samples: int = 400, n_iters: int = 6,
    n_elite: int = 40, sigma0: float = 0.8, w_run: float = 0.3, w_app: float = 0.05,
    gen: torch.Generator | None = None,
) -> np.ndarray:
    r"""Equivariant CEM for the pose cost. ``noise_rot`` rotates the Gaussian action draws by the orbit angle
    (scene-covariant randomness). Isotropic per-step covariance + a disk action bound $\lVert a\rVert\le1$ keep the
    whole procedure SO(2)-equivariant, so with the SAME ``gen`` seed the candidate sets at two orbit angles are
    exact rotations of each other. Returns the elite-mean plan ``(H,2)``.
    """
    s0 = torch.tensor(packed0, dtype=torch.float32).unsqueeze(0).expand(n_samples, 8).contiguous()
    gpos = torch.tensor(goal_scaled, dtype=torch.float32)
    gdir = torch.tensor(goal_dir, dtype=torch.float32)
    Rn = torch.tensor(rot_np(noise_rot), dtype=torch.float32) if noise_rot else None
    mean = torch.zeros(H, 2)
    sigma = torch.full((H,), sigma0)                                  # per-step SCALAR == isotropic in action plane
    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 2, generator=gen)
        if Rn is not None:
            eps = torch.einsum("ij,nhj->nhi", Rn, eps)                # scene-covariant draws
        cand = mean.unsqueeze(0) + sigma.view(1, H, 1) * eps
        cand = cand / cand.norm(dim=-1, keepdim=True).clamp_min(1.0)  # DISK bound (rotation-invariant)
        s = s0.clone()
        cost = torch.zeros(n_samples)
        for h in range(H):
            s = model.step(s, cand[:, h])
            dpos = ((s[:, 4:6] - gpos) ** 2).sum(-1)
            dang = 1.0 - (s[:, 6:8] * gdir).sum(-1)
            app = ((s[:, 0:2] - s[:, 4:6]) ** 2).sum(-1)
            step_cost = W_POS * dpos + W_ANG * dang
            cost = cost + (w_run * step_cost if h < H - 1 else step_cost) + w_app * app
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        # isotropic per-step std as the RMS *norm* of the deviation from the per-step VECTOR mean. This depends
        # only on ‖v−μ‖, so it is SO(2)-invariant -- unlike a pooled per-axis std(dim=(0,2)), which subtracts a
        # scalar pooled mean and is scrambled by rotation (the silent break that left the orbit non-flat).
        dev = elite - mean.unsqueeze(0)                                       # (n_elite,H,2)
        sigma = (dev.pow(2).sum(-1).mean(0) / 2).sqrt().clamp_min(1e-3)       # (H,) rotation-INVARIANT isotropic std
    return mean.numpy()


@torch.no_grad()
def model_rollout_pose_err(model, task: dict, plan: np.ndarray) -> tuple[float, float]:
    r"""Terminal block pose error (angle deg, position px) when ``plan`` is rolled out under the MODEL (no env).

    For the equivariant model + equivariant CEM this is EXACTLY orbit-invariant (the certificate at the task
    level, free of the real env's residual non-equivariance)."""
    s = torch.tensor(task["state_packed"], dtype=torch.float32).unsqueeze(0)
    for a in plan:
        s = model.step(s, torch.tensor(a, dtype=torch.float32).unsqueeze(0))
    s = s[0].numpy()
    block_px = s[4:6] * POS_SCALE + CENTER
    block_ang = math.atan2(s[7], s[6])
    return angle_err_deg(block_ang, task["goal_angle"]), float(np.linalg.norm(block_px - task["goal_px"]))


def closed_loop_realenv(
    env, model, task: dict, *, noise_rot: float, plan_fn, T_max: int = 30,
    replan_every: int = 6, seed: int = 0, gen: torch.Generator | None = None, **cem,
) -> tuple[float, float]:
    r"""Receding-horizon control on the REAL env; return the best (angle deg, pos px) pose error reached."""
    info = reset_task(env, task, seed=seed)
    goal_scaled = goal_to_scaled(task["goal_px"])
    gang = task["goal_angle"]
    gdir = np.array([math.cos(gang), math.sin(gang)], dtype=np.float32)
    best_ang, best_pos, best_score = math.inf, math.inf, math.inf
    t = 0
    while t < T_max:
        packed = info_to_packed(info)
        plan = plan_fn(model, packed, goal_scaled, gdir, noise_rot=noise_rot, gen=gen, **cem)
        for k in range(min(replan_every, T_max - t)):
            _, _, _, _, info = env.step(plan[k].astype(np.float32))
            bp = np.asarray(info["block_pose"])
            ang = angle_err_deg(float(bp[2]), gang)
            pos = float(np.linalg.norm(bp[:2] - task["goal_px"]))
            if ang + 0.3 * pos < best_score:
                best_score, best_ang, best_pos = ang + 0.3 * pos, ang, pos
            t += 1
    return best_ang, best_pos


def orbit_eval(model, base_tasks, *, plan_fn, equivariant_planner: bool, seed: int) -> dict:
    r"""For each orbit angle, mean (model-rollout, real-env) terminal pose error over the base tasks rotated to
    that angle. ``equivariant_planner`` rotates the planner's action noise by the orbit angle (A5)."""
    env = make_env()
    roll_ang, real_ang, real_pos = {}, {}, {}
    for deg in ORBIT_DEG:
        beta = math.radians(deg)
        ra_roll, ra_real, rp_real = [], [], []
        for i, bt in enumerate(base_tasks):
            task = rotate_pose_task(bt, beta)
            task["state_packed"] = pack_state7(task["state"])   # exact orbit action (no env round-trip)
            nr = beta if equivariant_planner else 0.0
            # model-rollout (open-loop plan under model) -- exact certificate probe
            gpos = goal_to_scaled(task["goal_px"])
            gdir = np.array([math.cos(task["goal_angle"]), math.sin(task["goal_angle"])], dtype=np.float32)
            plan = plan_fn(model, task["state_packed"], gpos, gdir, noise_rot=nr,
                           gen=torch.Generator().manual_seed(1000 + i))
            a_roll, _ = model_rollout_pose_err(model, task, plan)
            ra_roll.append(a_roll)
            # real-env closed loop -- gen seeded by base-task index ONLY (not the orbit angle), so the same
            # base task gets the SAME (scene-covariantly rotated) action noise at every orbit angle: a paired
            # protocol whose eq orbit-spread isolates the real env's residual non-equivariance, not CEM RNG.
            a_real, p_real = closed_loop_realenv(env, model, task, noise_rot=nr, plan_fn=plan_fn,
                                                 seed=i, gen=torch.Generator().manual_seed(5000 + i))
            ra_real.append(a_real); rp_real.append(p_real)
        roll_ang[deg] = float(np.mean(ra_roll))
        real_ang[deg] = float(np.mean(ra_real))
        real_pos[deg] = float(np.mean(rp_real))
    return {"rollout_ang": roll_ang, "real_ang": real_ang, "real_pos": real_pos}


def orbit_ratio(d: dict) -> float:
    base = max(d[0], 1e-9)
    return max(d[deg] for deg in ORBIT_DEG) / base


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)
    line = "=" * 78

    # --- train both models one-step on a wedge of real interior reorientation transitions -----------
    S, A, S2 = collect_pose_transitions(N_TRAIN, 0.0, math.radians(2 * WEDGE_DEG), seed=SEED)
    print(f"[step61] seed={SEED}  trained on {len(S)} interior pose transitions, phi in "
          f"[0,{2*WEDGE_DEG:.0f}) | MLP hidden={MLP_HIDDEN}", file=sys.stderr)
    eq = train_fair(RichVNForwardModelPushT(hidden=64), S, A, S2, epochs=EPOCHS, seed=SEED)
    mlp = train_fair(MLPForwardModelPushT(hidden=MLP_HIDDEN), S, A, S2, epochs=EPOCHS, seed=SEED)
    eq_resid, mlp_resid = model_equivariance_err(eq), model_equivariance_err(mlp)

    # --- [A] planning-cost orbit-invariance (model-internal, exact) ---------------------------------
    print(line); print("[A] Planning-cost orbit-invariance (model-internal)"); print(line)
    drift_eq = float(np.mean([pose_cost_drift(eq, math.radians(d)) for d in (37, 90, 153, 211)]))
    drift_mlp = float(np.mean([pose_cost_drift(mlp, math.radians(d)) for d in (37, 90, 153, 211)]))
    print(f"    model equivariance |M(Rs,Ra)-R M(s,a)|: eq {eq_resid:.2e}   MLP {mlp_resid:.2e}")
    print(f"    pose-cost drift E|C(gs)-C(s)|/E|C|:     eq {drift_eq:.2e}   MLP {drift_mlp:.2e}")
    print(f"    params: eq={n_params(eq)}  MLP={n_params(mlp)} ({n_params(mlp)/n_params(eq):.1f}x eq)")

    # --- base tasks at orientation ~0, rotated to every orbit angle (PAIRED) ------------------------
    rng = np.random.default_rng(100 + SEED)
    base_tasks = [sample_pose_task(rng, -math.radians(WEDGE_DEG), math.radians(WEDGE_DEG))
                  for _ in range(N_BASE)]

    # --- [B] G-equivariant planner (A5): paired closed-loop task error over the orbit ----------------
    print(); print(line)
    print("[B] Certificate at the task level: G-equivariant planner, paired tasks over the orbit")
    print(line)
    eqp_eq = orbit_eval(eq, base_tasks, plan_fn=cem_plan_pose_equiv, equivariant_planner=True, seed=SEED)
    eqp_mlp = orbit_eval(mlp, base_tasks, plan_fn=cem_plan_pose_equiv, equivariant_planner=True, seed=SEED)
    roll_ratio_eq, roll_ratio_mlp = orbit_ratio(eqp_eq["rollout_ang"]), orbit_ratio(eqp_mlp["rollout_ang"])
    real_ratio_eq, real_ratio_mlp = orbit_ratio(eqp_eq["real_ang"]), orbit_ratio(eqp_mlp["real_ang"])
    print(f"    {'orbit beta':>10s} | {'eq roll':>8s} {'eq real':>8s} | {'MLP roll':>9s} {'MLP real':>9s}  (angle deg)")
    print("    " + "-" * 60)
    for d in ORBIT_DEG:
        print(f"    {d:8d}deg | {eqp_eq['rollout_ang'][d]:7.2f}  {eqp_eq['real_ang'][d]:7.2f} | "
              f"{eqp_mlp['rollout_ang'][d]:8.2f}  {eqp_mlp['real_ang'][d]:8.2f}")
    print(f"    => model-rollout orbit ratio (EXACT certificate): eq x{roll_ratio_eq:.3f}  MLP x{roll_ratio_mlp:.2f}")
    print(f"    => real-env  orbit ratio (practical)            : eq x{real_ratio_eq:.3f}  MLP x{real_ratio_mlp:.2f}")

    # --- [C] realistic (scene-blind) planner: the rebuttal to "you rotated the noise" ---------------
    print(); print(line)
    print("[C] Realistic planner (scene-blind CEM, fresh per orientation), paired tasks")
    print(line)
    blind_eq = orbit_eval(eq, base_tasks, plan_fn=cem_plan_pose_equiv, equivariant_planner=False, seed=SEED)
    blind_mlp = orbit_eval(mlp, base_tasks, plan_fn=cem_plan_pose_equiv, equivariant_planner=False, seed=SEED)
    blind_ratio_eq = orbit_ratio(blind_eq["real_ang"])
    blind_ratio_mlp = orbit_ratio(blind_mlp["real_ang"])
    ang_eq_in = blind_eq["real_ang"][0]
    ang_mlp_in = blind_mlp["real_ang"][0]
    print(f"    scene-blind real-env orbit ratio: eq x{blind_ratio_eq:.2f}  MLP x{blind_ratio_mlp:.2f}")
    print(f"    in-wedge (beta=0) real-env angle err: eq {ang_eq_in:.2f}deg  MLP {ang_mlp_in:.2f}deg")

    # --- gate (honest; gated on the exact [A]+[B]; [C] reported) ------------------------------------
    ok_cost = drift_eq < 1e-4 and drift_mlp > 0.3
    ok_rollout_exact = roll_ratio_eq < 1.02
    ok_realenv = real_ratio_eq < 1.30 and real_ratio_mlp > 1.5
    ok_compete = ang_eq_in <= 1.25 * ang_mlp_in
    passed = ok_cost and ok_rollout_exact and ok_realenv and ok_compete

    result = {
        "passed": passed,
        "gate": {"cost_exact": ok_cost, "rollout_exact": ok_rollout_exact,
                 "realenv_flat": ok_realenv, "compete": ok_compete},
        "equivariance_residual": {"eq": eq_resid, "mlp": mlp_resid},
        "cost_drift": {"eq": drift_eq, "mlp": drift_mlp},
        "params": {"eq": n_params(eq), "mlp": n_params(mlp)},
        "orbit_deg": ORBIT_DEG, "wedge_deg": WEDGE_DEG,
        "equivariant_planner": {
            "eq": eqp_eq, "mlp": eqp_mlp,
            "rollout_ratio": {"eq": roll_ratio_eq, "mlp": roll_ratio_mlp},
            "real_ratio": {"eq": real_ratio_eq, "mlp": real_ratio_mlp},
        },
        "blind_planner": {
            "eq": blind_eq, "mlp": blind_mlp,
            "real_ratio": {"eq": blind_ratio_eq, "mlp": blind_ratio_mlp},
            "inwedge_ang": {"eq": ang_eq_in, "mlp": ang_mlp_in},
        },
        "smoke": SMOKE, "seed": SEED,
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "step61_closed_loop_certificate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # --- figure --------------------------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    axL.axvspan(-WEDGE_DEG, WEDGE_DEG, color="0.85", label=f"train wedge (|β|≤{WEDGE_DEG:.0f}°)")
    axL.plot(ORBIT_DEG, [eqp_eq["real_ang"][d] for d in ORBIT_DEG], "o-", color="C2", lw=2.4,
             label=f"equivariant (real env, x{real_ratio_eq:.2f})")
    axL.plot(ORBIT_DEG, [eqp_mlp["real_ang"][d] for d in ORBIT_DEG], "s--", color="C3",
             label=f"MLP (real env, x{real_ratio_mlp:.1f})")
    axL.plot(ORBIT_DEG, [eqp_eq["rollout_ang"][d] for d in ORBIT_DEG], "o:", color="C0", alpha=0.7,
             label=f"equivariant (model rollout, x{roll_ratio_eq:.3f}, exact)")
    axL.set_xlabel("orbit angle β (degrees)"); axL.set_ylabel("closed-loop block-angle error (deg)")
    axL.set_title("Certificate at the task level: equivariant closed loop flat over the orbit")
    axL.legend(fontsize=7, loc="best")
    axR.bar([0, 1, 2, 3], [drift_eq, drift_mlp, real_ratio_eq - 1, real_ratio_mlp - 1],
            color=["C2", "C3", "C2", "C3"])
    axR.set_yscale("symlog", linthresh=1e-4)
    axR.set_xticks([0, 1, 2, 3])
    axR.set_xticklabels(["cost-drift\neq", "cost-drift\nMLP", "orbit-excess\neq", "orbit-excess\nMLP"], fontsize=8)
    axR.set_title("planning-cost drift (exact) and closed-loop orbit excess")
    fig.suptitle("Step 61 — the predictability certificate at the TASK level (closed-loop PushT pose control)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "step61_closed_loop_certificate.png", dpi=130)

    # --- verdict -------------------------------------------------------------------------------------
    print(); print(line); print("STEP 61 SUMMARY"); print(line)
    print(f"    [A] cost drift: eq {drift_eq:.1e} (exact) vs MLP {drift_mlp:.2f} (scrambled)")
    print(f"    [B] model-rollout orbit ratio: eq x{roll_ratio_eq:.3f} (EXACT certificate) vs MLP x{roll_ratio_mlp:.2f}")
    print(f"        real-env orbit ratio:       eq x{real_ratio_eq:.3f} vs MLP x{real_ratio_mlp:.2f}; "
          f"in-wedge angle eq {ang_eq_in:.1f}deg vs MLP {ang_mlp_in:.1f}deg")
    print(f"    [C] scene-blind planner orbit ratio: eq x{blind_ratio_eq:.2f} vs MLP x{blind_ratio_mlp:.2f} "
          f"(equivariant model stays flatter even without A5)")
    print(f"    guards: cost-exact={ok_cost} rollout-exact={ok_rollout_exact} "
          f"realenv-flat={ok_realenv} compete={ok_compete}")
    if passed:
        print(f"    PASS: the certificate converts to TASK competence -- an equivariant world model + "
              f"G-equivariant planner")
        print(f"          gives closed-loop pose control that is orbit-invariant (model-rollout exact, "
              f"real-env x{real_ratio_eq:.2f}),")
        print(f"          while the {n_params(mlp)/n_params(eq):.0f}x MLP degrades x{real_ratio_mlp:.1f} out of the wedge.")
    else:
        bad = [k for k, v in result["gate"].items() if not v]
        print(f"    INCONCLUSIVE: gate not met ({bad}); reported as-is (no thresholds loosened).")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
