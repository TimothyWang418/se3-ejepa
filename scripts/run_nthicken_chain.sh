#!/bin/bash
# n-thickening overnight chain v2 (2026-06-11): R4 -> R3 -> R2 -> R1, all CPU, polite (nice + half-cores).
# GPU untouched. Each stage logs to /tmp/, milestones to /tmp/chain.log. Stages independent (no && chaining).
cd ~/se3-ejepa
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
P=".venv/bin/python"
echo "CHAIN-START $(date)" >> /tmp/chain.log

# R4: kappa estimator densified (41 samples, longer windows/orbits)
STEP95_SAMPLES=41 STEP95_WB=400 STEP95_TB=1300 STEP95_WC=200 STEP95_TC=520 \
  nice -n 10 $P experiments/step95_prefactor_angles.py > /tmp/r4_step95.log 2>&1
echo "R4-DONE $(date)" >> /tmp/chain.log

# R3: monitor episodes 20 -> 100
STEP94_EPISODES=100 nice -n 10 $P experiments/step94_budgeted_monitor.py > /tmp/r3_step94.log 2>&1
STEP96_EPISODES=100 nice -n 10 $P experiments/step96_taxonomy_monitor.py > /tmp/r3_step96.log 2>&1
echo "R3-DONE $(date)" >> /tmp/chain.log

# R2: audit expansion (up to 69 new cells; incremental+resumable)
nice -n 10 $P experiments/step89b_audit_expansion.py > /tmp/r2_step89b.log 2>&1
echo "R2-DONE $(date)" >> /tmp/chain.log

# R1: E12 flagship seeds 3..19, one seed per invocation (resumable); outputs renamed per seed, canonical restored
for s in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  STEP85_SEEDS=$s nice -n 10 $P experiments/step85_trustworthy_cert_downstream.py > /tmp/r1_step85_seed$s.log 2>&1
  for f in $(git diff --name-only -- 'papers/figures/step85_*.json'); do
    b=$(basename "$f" .json); mv "$f" "papers/figures/${b}_seed${s}.json"; git checkout -q "$f"
  done
  echo "R1-seed$s-DONE $(date)" >> /tmp/chain.log
done
echo "R1-DONE $(date)" >> /tmp/chain.log
echo "ALL-DONE $(date)" >> /tmp/chain.log
