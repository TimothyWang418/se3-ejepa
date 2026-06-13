"""Pod-side audit of a trained bridge checkpoint (eq or dense TD-MPC2 WorldModel) — the step89/step100
machinery on the policy-prior loop g(z)=next(z, tanh(pi(z))). Run after training; writes a tiny JSON.

Usage (on pod, after a run finishes):
  cd /root/tdmpc2/tdmpc2 && MUJOCO_GL=egl python /root/bridge_audit.py <eq|dense> <seed> <ckpt_path>
Writes /root/audit_<arm>_s<seed>.json  (lambda1 + CI + measured ratio@eps).
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from common.world_model import WorldModel
from common.eq_world_model import make_eq_world_model
from envs import make_env

ARM, SEED, CKPT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
EPS = [0.05, 0.1, 0.2]

# rebuild cfg from the run's hydra output, else minimal walker cfg
cfg = OmegaConf.create(dict(
    multitask=False, tasks=['walker-walk'], task_dim=0, obs='state', obs_shape={'state':[24]},
    action_dim=6, action_dims=[6], num_enc_layers=2, enc_dim=256, latent_dim=512, simnorm_dim=8,
    mlp_dim=512, num_bins=101, num_q=5, dropout=0.01, episodic=False, log_std_min=-10.0, log_std_max=2.0,
    tau=0.01, vmin=-10.0, vmax=10.0, num_channels=32, seed=SEED))
cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
WM = make_eq_world_model(WorldModel) if ARM == 'eq' else WorldModel
model = WM(cfg).to(dev)
sd = torch.load(CKPT, map_location=dev)
model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
model.eval()

@torch.no_grad()
def enc(o): return model.encode(torch.as_tensor(o, dtype=torch.float32, device=dev).unsqueeze(0), None)
def g(z):  # policy-prior closed loop
    a, _ = model.pi(z, None); return model.next(z, torch.tanh(a) if a.abs().max()>1 else a, None)

# collect a true episode under the policy for z0 + measured rollouts
env = make_env(cfg); ts = env.reset(); obs=[ts[0] if isinstance(ts,tuple) else ts]
o = obs[0]
for _ in range(420):
    with torch.no_grad():
        a,_ = model.pi(enc(o), None)
    step = env.step(a.squeeze(0).cpu().numpy())
    o = step[0]; obs.append(o)
obs = np.array([x for x in obs], dtype=np.float32)
Z = torch.cat([enc(o) for o in obs])
z0 = Z[len(Z)//2:len(Z)//2+1]
D = z0.shape[1]
# Benettin on leading k dirs
k, steps, warm = 12, 200, 40
torch.manual_seed(1000+SEED); Q = torch.linalg.qr(torch.randn(D,k,device=dev))[0]
z=z0.clone(); sums=torch.zeros(k,device=dev); hist=[]
for t in range(steps+warm):
    J = torch.autograd.functional.jacobian(lambda zz: g(zz.unsqueeze(0))[0], z[0])
    Q = J@Q; Q,R = torch.linalg.qr(Q); d=R.diagonal().abs().clamp_min(1e-30)
    if t>=warm: sums+=d.log(); hist.append(float(d[0].log()))
    z=g(z)
    if not torch.isfinite(z).all(): break
lam1=float((sums/steps)[0]); H=np.array(hist)
bl=np.array_split(H,20); rng=np.random.default_rng(7)
boot=[np.mean(rng.choice([b.mean() for b in bl],20)) for _ in range(500)]
lo,hi=float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))
# measured crossings
scale=float(Z.std()); starts=np.linspace(0,len(obs)-130,80).astype(int); meas={str(e):[] for e in EPS}
with torch.no_grad():
    for s0 in starts:
        z=Z[s0:s0+1]; cr={str(e):None for e in EPS}
        for h in range(1,120):
            z=g(z)
            if s0+h>=len(obs): break
            err=float((z-Z[s0+h]).norm())/(float(Z[s0+h].norm())+1e-9)
            for e in EPS:
                if cr[str(e)] is None and err>e: cr[str(e)]=h
        for e in EPS: meas[str(e)].append(cr[str(e)] or 120)
rows=[]
for e in EPS:
    med=float(np.median(meas[str(e)])); t1=float(np.log(1/e)/lam1) if (lam1>0 and lo>0) else None
    rows.append({"eps":e,"T1":t1,"measured_median":med,"ratio":(med/t1) if t1 else None})
out={"arm":ARM,"seed":SEED,"lambda1":lam1,"lambda1_ci":[lo,hi],"rows":rows,"latent_scale":scale}
open(f"/root/audit_{ARM}_s{SEED}.json","w").write(json.dumps(out,indent=1))
r02=[r for r in rows if r['eps']==0.2][0]
print(f"[audit] {ARM}-s{SEED}: lam1={lam1:+.4f} CI[{lo:+.3f},{hi:+.3f}] med@0.2={r02['measured_median']:.0f} ratio={('%.2f'%r02['ratio']) if r02['ratio'] else '—'}")
