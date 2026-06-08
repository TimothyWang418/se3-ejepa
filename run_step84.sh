#!/usr/bin/env bash
# Step 84 — the SPOTLIGHT certified-control-benchmark triad on Gymnasium Acrobot-v1.
#
# One command. The RTX 3080 (CUDA) box runs THIS to scale up the experiment, write the figure, and commit + push the
# results back. The Mac CPU-smoke-tested the whole pipeline first (the 3080 run is the scale-up, not the debug).
#
#   bash run_step84.sh
#
# It (1) prints torch.cuda.is_available(), (2) ensures `gymnasium[classic_control]` is installed (pure-python classic
# control, no MuJoCo / Vulkan), (3) runs the FULL experiment (writes papers/figures/step84_certified_control_benchmark
# .{png,json}), then (4) stages step84 BY NAME, commits with a results HEREDOC, and pushes.
#
# Device-agnostic: the experiment auto-selects cuda > mps > cpu. WM train + CEM rollouts are float32 on the device; the
# spectral/Jacobian readout is float64 on CPU regardless (the 3080's fp64 is gimped). Override the python if needed:
#   PY=python bash run_step84.sh
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
GREEN='\033[0;32m'; RED='\033[0;31m'; YEL='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[ok]${NC}   $1"; }
warn() { echo -e "  ${YEL}[warn]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }

echo "== step84 certified-control-benchmark triad (Acrobot-v1, Z_2) =="

# 1. device report (the spotlight runs on the 3080; falls back to cpu/mps anywhere else)
"$PY" - <<'PYEOF'
import torch
print(f"    torch {torch.__version__}; cuda_available={torch.cuda.is_available()}; "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (cpu/mps)'}")
PYEOF

# 2. ensure gymnasium[classic_control] (pure-python classic control; no MuJoCo). Idempotent.
if "$PY" - <<'PYEOF' 2>/dev/null
import gymnasium as gym
gym.make("Acrobot-v1").close()
PYEOF
then ok "gymnasium[classic_control] present (Acrobot-v1 makes)"
else
  warn "Acrobot-v1 unavailable — installing gymnasium[classic_control]"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -q 'gymnasium[classic_control]' && ok "installed gymnasium[classic_control] (uv)" \
      || fail "uv pip install gymnasium[classic_control] failed"
  else
    "$PY" -m pip install -q 'gymnasium[classic_control]' && ok "installed gymnasium[classic_control] (pip)" \
      || fail "pip install gymnasium[classic_control] failed"
  fi
fi

# 3. run the FULL experiment (writes the figure + JSON). An INCONCLUSIVE triad is a valid scientific outcome — the
#    experiment returns 0 unless an actual ERROR occurs, so we let it run and commit whatever verdict lands.
echo "-- running the full triad (this is the scale-up; minutes-to-tens-of-minutes on the 3080) --"
"$PY" experiments/step84_certified_control_benchmark.py
RC=$?
if [ $RC -ne 0 ]; then fail "experiment exited non-zero ($RC) — NOT committing"; exit $RC; fi
ok "experiment finished (exit 0); figure + JSON written"

# 4. stage step84 BY NAME, commit with a results HEREDOC, push (the 3080 commits + pushes so the controller can pull).
FIG="papers/figures/step84_certified_control_benchmark"
git add experiments/step84_certified_control_benchmark.py tests/test_step84.py run_step84.sh "${FIG}.png" "${FIG}.json"

# pull a one-line verdict + the headline numbers out of the JSON for the commit message
SUMMARY=$("$PY" - <<'PYEOF'
import json, pathlib
p = pathlib.Path("papers/figures/step84_certified_control_benchmark.json")
try:
    d = json.loads(p.read_text())
except Exception as e:
    print(f"verdict: (could not read JSON: {e})"); raise SystemExit(0)
eq = d.get("equivariant", {}) or {}
cert = eq.get("cert", {}) or {}
g0 = eq.get("G0", {}) or {}
gb = eq.get("G_binding", {}) or {}
gii = eq.get("G_ii", {}) or {}
print(f"verdict: {d.get('verdict','?')}")
print(f"equi WM: lambda1={cert.get('lambda1'):.4f} CI={cert.get('lambda1_ci')} "
      f"T1_steps={cert.get('T1_steps')} route={cert.get('route')} relMSE={eq.get('one_step_relmse'):.3e}")
print(f"gates(equi): G0={g0.get('passed')} (CI-strong={g0.get('strong')})  "
      f"G-binding={gb.get('passed')} (H*={gb.get('H_star')}, spread={gb.get('spread'):.1f}, flat={gb.get('flat')})  "
      f"G-ii={gii.get('passed')}")
ti = (eq.get('triad_i') or {}).get('rows', [])
for r in ti:
    print(f"  eps={r['eps']}: certified T1={r['T1_steps']} vs measured med={r['measured_median']:.0f} "
          f"(ratio {r['ratio_measured_over_certified']:.2f}, route {r['route']})")
neq = d.get("non_equivariant", {}) or {}
if neq:
    nc = neq.get("cert", {}) or {}; ng = neq.get("G_binding", {}) or {}; ngii = neq.get("G_ii", {}) or {}
    print(f"non-equi WM (contrast): lambda1={nc.get('lambda1'):.4f} T1_steps={nc.get('T1_steps')} "
          f"G-binding={ng.get('passed')} G-ii={ngii.get('passed')}")
PYEOF
)
echo "$SUMMARY"

git commit -F- <<EOF
step84: certified-control-benchmark triad on Acrobot-v1 (Z_2) — 3080 results

Spotlight bet: certified predictability horizon read off a LEARNED model of a
recognized chaotic CONTROL benchmark (Gymnasium Acrobot-v1, Z_2 reflection),
(i) accurate vs measured rollout-divergence, (ii) actionable (plan-depth-gated
CEM-MPC vs blind, ablated to the certificate), (iii) on an env where the
horizon is the binding constraint. Honest gates G0 / G-binding / G-ii; an
INCONCLUSIVE verdict (flat-in-H, the step79-D1 mode) is reported, never loosened.

${SUMMARY}

Equivariant WM = Z_2 frame-averaging (step81 pattern); non-equivariant baseline
reported for contrast. Control-setting certificate = finite-time lambda1 of the
product of action-conditioned state-Jacobians along a swing-up rollout (step78
QR + block-bootstrap; step82 cone where it certifies, bootstrap otherwise), in
ENV STEPS. CPU-smoke-tested end-to-end on Mac before scale-up.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
ok "committed step84 results"

git push && ok "pushed" || fail "git push failed (push manually so the controller can pull)"
echo "== done =="
