# ⛔ DEPRECATED (2026-06-13, after a 4090 attempt) — DO NOT USE FOR eq TRAINING

**Second rental attempt (RTX 4090) FAILED — different cause than B300.** e2cnn 0.2.3 installed
and the eq FORWARD was numerically correct (equivariance held, z-norms matched, no NaN). But eq
TRAINING DIVERGED on the pod's torch 2.8+cu128: champion runs hit latent-std 17 / 0.4 (vs the
stable ~1 on Mac/box torch 2.12+cu130). Root cause: **e2cnn training numerics are torch-version-
sensitive, and the Mac/box's torch 2.12+cu130 is NOT installable on a rental** (pytorch cu124
index maxes at 2.6; torch 2.6 then broke torchvision). Switching GPU cards does NOT help — the
bottleneck is the torch build, not the hardware.

**THE COMBINED VERDICT (two rentals, two distinct failures):** our research stack — small models
+ e2cnn equivariance + a specific bleeding-edge torch + CPU audits — is HIGHLY LOCALIZED. Cloud
rentals essentially have NO use case for us: B300 was workload-mismatch (memory/knn, FLOPs
unused), 4090 was numerics-non-portability (eq training diverges off the exact torch). **Rule:
do NOT rent again unless (a) the workload is genuinely FLOP-bound AND (b) the EXACT torch+cuda
build of the reference machine is installable on the rental AND (c) eq TRAINING (not just
forward) is verified numerically stable there.** C3 n=30 stays on the free Mac (slow but correct
and comparable). The runbook below is kept only as a record of the (failed) setup.

---

# Rental package — 2D pixel-eq high-n armoring (ready-to-fire; B300 lessons baked in)

*Prepared 2026-06-13. The ONE workload that genuinely benefits from a rental: the 2D pixel-eq
armoring at high n (e2cnn steerable convs are FLOP-heavy, eq ≈ 4× plain; C3 n=30 ≈ 20 h on the
free Mac MPS → ~2-3 h on a rented 4090/A100). Independent of the 3D lane — can run in parallel
with the free 3080 3D work. Fire when you want the publication-grade volume; ~$5-10.*

## What it buys (the experiments)
- **C3 n=30** (champion eq + plain × c2000): tightens the guarantee CI — the paper's strongest
  claim. `P4_N=30`. (was n=10 → 90 qualifying audits; n=30 → ~270.)
- **v1.6 recipe grid n=10** (stability-content tradeoff, publication n).
- **Finer cliff / eq cliff at larger n** (figure-grade P(fail), already partly done free).
- **T1.9 192px tuned-recipe re-test** (closes the off-tune caveat on claim 10).

## GPU pick
**RTX 4090 (24GB)** — cheapest (~$0.4-0.7/h), 24GB clears the VN-style swap entirely, e2cnn fine.
A100 only if running everything at once. NOT a B300 (overkill, mismatched — see post-mortem).

## Bootstrap (one paste, on the rental, after SSH-over-exposed-TCP works)
```bash
# 0. deps — the VERIFIED working combo (torch 2.x + e2cnn 0.2.3; B300 failed only from rushed setup)
pip install --break-system-packages torch e2cnn==0.2.3 pygame pymunk opencv-python-headless \
    gymnasium scipy shapely loguru numpy
# 1. repo (git clone the remote OR runpodctl from Mac) into /root/se3-ejepa  (NOT /workspace —
#    MooseFS breaks rsync temp-rename; /root local disk is fast)
cd /root/se3-ejepa && pip install --break-system-packages -e third_party/stable-worldmodel
# verify e2cnn + swm
python3 -c "import torch,e2cnn,pygame,pymunk,stable_worldmodel as s; import gymnasium as gym; \
    gym.make('swm/PushT-v1').reset(seed=0); print('READY cuda', torch.cuda.is_available())"
```

## Data transfer — runpodctl P2P, NOT the SSH proxy (proxy throttles 0.4 MB/s; P2P = 12 MB/s)
```bash
# on Mac: tar the two corpora, send (gives a code)
cd ~/Workspace/se3-ejepa/data/p4_step1 && tar cf /tmp/c2d.tar corpus_c2000.npz wedge_corpus.npz wedge_ho_in.npz
/tmp/runpodctl send /tmp/c2d.tar            # prints: runpodctl receive <code>
# on rental: receive + unpack to the data dir
cd /root/se3-ejepa/data/p4_step1 && runpodctl receive <code> && tar xf c2d.tar && rm c2d.tar
```

## Launch (tmux — nohup/setsid did NOT detach on the box; tmux is the reliable mechanism)
```bash
# window 1: C3 n=30 (the headline)
tmux new-session -d -s c3 "cd /root/se3-ejepa && PYTHONPATH=/root/se3-ejepa P4_DEVICE=cuda \
    P4_N=30 P4_TAG=_rental python3 -u experiments/p4_champion_confirm.py > /root/c3.log 2>&1"
# window 2 (if A100/parallel-safe): v1.6 recipe grid — else run after C3
tmux new-session -d -s rec "cd /root/se3-ejepa && PYTHONPATH=/root/se3-ejepa P4_DEVICE=cuda \
    python3 -u experiments/p4_v16_stageA_sweep.py > /root/rec.log 2>&1"
```
*Single GPU: eq runs are GPU-bound → 2-3 parallel max before contention (the VN lesson). On a
4090, run C3 first (sequential n=30), then the grid. On an A100 (40-80GB) run both windows.*

## Pull results + release
```bash
# on rental: tar artifacts, send
cd /root/se3-ejepa/papers/figures && tar cf /tmp/out.tar p4_champion_confirm_rental.json p4_v16_*.json
runpodctl send /tmp/out.tar
# on Mac: receive, ledger, THEN terminate the pod (not just stop — stop still bills storage)
```

## Lessons baked in (from the B300 post-mortem + the VN/orchestration debugging)
1. Confirm the bottleneck is GPU FLOPs before renting. 2D eq IS (steerable convs). 3D VN is NOT
   (memory; fixed on the free 3080). Audits/corpus are CPU — never rent for them.
2. `runpodctl send/receive` for data (P2P, 12 MB/s), never the SSH proxy (0.4 MB/s, drops big files).
3. `/root` local disk, not `/workspace` (MooseFS breaks rsync).
4. `tmux` for persistence (nohup/setsid did not detach reliably; setsid procs need `kill -9 PID`).
5. Watch peak VRAM; keep batch so peak < card VRAM (no swap — that was the "20s/batch" killer).
6. One-at-a-time or ≤3 parallel for GPU-bound runs (over-parallel contends, self-defeating).
7. Use a FRESH log filename per launch (stale logs masked the real state during debugging).
8. Terminate (not Stop) the pod when done — Stop still bills the volume.
