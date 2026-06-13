#!/bin/bash
SEED=${1:-0}
python - <<'PYEOF'
p="/root/tdmpc2/tdmpc2/trainer/online_trainer.py"; s=open(p).read()
anc="\t\t\teval_next = True\n"
ins="\t\t\tif self._step>0 and self._step%10000==0:\n\t\t\t\ttry: self.agent.save(f'/root/latest_{self.cfg.model}_s{self.cfg.seed}.pt')\n\t\t\t\texcept Exception as e: print('save err',e)\n"
if "latest_{self.cfg.model}" not in s:
    s=s.replace(anc, anc+ins, 1)
open(p,"w").write(s); print("trainer patched (save every 10k)")
PYEOF
pkill -9 -f "tra[i]n.py"; sleep 3
cd /root/tdmpc2/tdmpc2; export MUJOCO_GL=egl
for arm in eq dense; do
  setsid bash -c "python train.py task=walker-walk model=$arm seed=$SEED steps=200000 seed_steps=5000 eval_episodes=1 eval_freq=25000 exp_name=br3_${arm}_s$SEED enable_wandb=false save_csv=false save_video=false save_agent=true compile=true; echo ${arm}-s${SEED}-DONE:\$?" </dev/null >/root/br3_${arm}_s$SEED.log 2>&1 &
done
sleep 3; echo "relaunched seed $SEED: both compile=true, periodic-save 10k, target 200k"
