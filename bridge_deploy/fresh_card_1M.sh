#!/bin/bash
SEED=${1:-3}
cd /root/tdmpc2/tdmpc2
cp /root/eq_world_model.py common/eq_world_model.py
python - <<'PYEOF'
p="tdmpc2.py"; s=open(p).read()
old="\t\tself.model = WorldModel(cfg).to(self.device)"
if "make_eq_world_model" not in s and s.count(old)==1:
    new="\t\tif getattr(cfg,'model','dense')=='eq':\n\t\t\tfrom common.eq_world_model import make_eq_world_model\n\t\t\tself.model=make_eq_world_model(WorldModel)(cfg).to(self.device)\n\t\telse:\n\t\t\tself.model=WorldModel(cfg).to(self.device)"
    s=s.replace(old,new,1)
import re
s=s.replace('mode="reduce-overhead"','mode="default"')
open(p,"w").write(s)
c="config.yaml"; cy=open(c).read()
if "\nmodel:" not in cy: open(c,"a").write("\nmodel: dense\n")
print("model=eq flag + mode=default")
PYEOF
python - <<'PYEOF'
p="trainer/online_trainer.py"; s=open(p).read()
anc="\t\t\teval_next = True\n"
ins="\t\t\tif self._step>0 and self._step%25000==0:\n\t\t\t\ttry: self.agent.save(f'/root/latest_{self.cfg.model}_s{self.cfg.seed}.pt')\n\t\t\t\texcept Exception as e: print('save err',e)\n"
if "latest_{self.cfg.model}" not in s: s=s.replace(anc,anc+ins,1); open(p,"w").write(s)
print("save patch")
PYEOF
grep -q 'mode="default"' tdmpc2.py && echo "mode=default OK"
export MUJOCO_GL=egl
for arm in eq dense; do
  setsid bash -c "python train.py task=walker-walk model=$arm seed=$SEED steps=1000000 seed_steps=5000 eval_episodes=1 eval_freq=50000 exp_name=br4_${arm}_s$SEED enable_wandb=false save_csv=false save_video=false save_agent=true compile=true; echo ${arm}-s${SEED}-DONE:\$?" </dev/null >/root/br4_${arm}_s$SEED.log 2>&1 &
done
sleep 3; echo "launched seed $SEED @ 1M (eq+dense, mode=default, save 25k)"
