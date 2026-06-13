"""S2(leg-exchange)-equivariant TD-MPC2 WorldModel for the walker bridge (step101).

Deployment copy of experiments/step101_eq_world_model.py from the se3-ejepa repo (that file is the tested
reference; G-EQ passes at machine precision). Self-contained: uses tdmpc2's own common.layers so it drops
into the official trainer. Activated by `model=eq` in the cfg; `model=dense` (default) = stock WorldModel.

Walker single-task layout: obs 24, action 6, latent 512 (SimNorm groups of 8 = 64 groups), split as
32 invariant groups (256-d) | 16 R-groups (128-d) | 16 L-groups (128-d). Adapter forwards match the
official call sites: encoder(obs24), _dynamics(cat[z512,a6]=518), _reward(518), _pi(z512).
"""
import torch
import torch.nn as nn

from common import layers

OBS, ACT, LAT = 24, 6, 512
INV_D, LEG_D = 256, 128
T_IDX = [0, 1, 14, 15, 16, 17]
R_IDX = list(range(2, 8)) + list(range(18, 21))
L_IDX = list(range(8, 14)) + list(range(21, 24))
ACT_R, ACT_L = [0, 1, 2], [3, 4, 5]


def rho_obs(o):
    out = o.clone(); out[..., R_IDX], out[..., L_IDX] = o[..., L_IDX], o[..., R_IDX]; return out


def rho_act(a):
    out = a.clone(); out[..., ACT_R], out[..., ACT_L] = a[..., ACT_L], a[..., ACT_R]; return out


def rho_z(z):
    inv, r, l = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
    return torch.cat([inv, l, r], dim=-1)


def _mlp(cfg, i, o):
    return layers.mlp(i, [cfg.enc_dim], o)


class EqEncState(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.leg = layers.mlp(len(R_IDX) + len(T_IDX), [cfg.enc_dim], LEG_D)
        s.inv = layers.mlp(len(T_IDX) + LEG_D, [cfg.enc_dim], INV_D)
        s.sn = layers.SimNorm(cfg)

    def forward(s, o):
        t, r, l = o[..., T_IDX], o[..., R_IDX], o[..., L_IDX]
        zr, zl = s.leg(torch.cat([r, t], -1)), s.leg(torch.cat([l, t], -1))
        zi = s.inv(torch.cat([t, zr + zl], -1))
        return s.sn(torch.cat([zi, zr, zl], -1))


class EqDynamics(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.leg = layers.mlp(LEG_D + INV_D + LEG_D + 3 + 3, 2 * [cfg.mlp_dim], LEG_D)
        s.inv = layers.mlp(INV_D + LEG_D + 3, 2 * [cfg.mlp_dim], INV_D)
        s.sn = layers.SimNorm(cfg)

    def forward(s, x):                                  # x = cat([z 512, a 6]) = 518
        z, a = x[..., :LAT], x[..., LAT:LAT + ACT]
        zi, zr, zl = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
        ar, al = a[..., ACT_R], a[..., ACT_L]
        pool, apool = zr + zl, ar + al
        zr2 = s.leg(torch.cat([zr, zi, pool, ar, apool], -1))
        zl2 = s.leg(torch.cat([zl, zi, pool, al, apool], -1))
        zi2 = s.inv(torch.cat([zi, pool, apool], -1))
        return s.sn(torch.cat([zi2, zr2, zl2], -1))


class EqReward(nn.Module):                              # S2-invariant
    def __init__(s, cfg):
        super().__init__()
        s.net = layers.mlp(INV_D + LEG_D + 3, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))

    def forward(s, x):                                  # x = cat([z 512, a 6])
        z, a = x[..., :LAT], x[..., LAT:LAT + ACT]
        zi, zr, zl = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
        return s.net(torch.cat([zi, zr + zl, a[..., ACT_R] + a[..., ACT_L]], -1))


class EqPi(nn.Module):                                  # S2-equivariant: out = [meanR,meanL,logstdR,logstdL]
    def __init__(s, cfg):
        super().__init__()
        s.head = layers.mlp(LEG_D + INV_D, 2 * [cfg.mlp_dim], 6)

    def forward(s, z):
        zi, zr, zl = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
        hr, hl = s.head(torch.cat([zr, zi], -1)), s.head(torch.cat([zl, zi], -1))
        return torch.cat([hr[..., :3], hl[..., :3], hr[..., 3:], hl[..., 3:]], -1)


def make_eq_world_model(WorldModel):
    """Returns an EqWorldModel subclass bound to the official WorldModel (avoids import cycles)."""
    class EqWorldModel(WorldModel):
        def __init__(self, cfg):
            assert not cfg.multitask, "eq bridge is single-task walker"
            assert cfg.latent_dim == LAT and tuple(cfg.obs_shape['state']) == (OBS,) and cfg.action_dim == ACT, \
                "eq layout is hardcoded to walker-walk (obs24/act6/latent512)"
            super().__init__(cfg)
            self._encoder = nn.ModuleDict({'state': EqEncState(cfg)})
            self._dynamics = EqDynamics(cfg)
            self._reward = EqReward(cfg)
            self._pi = EqPi(cfg)
            from common import init
            self.apply(init.weight_init)
            init.zero_([self._reward.net[-1].weight])
            self.init()                                  # rebuild Q target/detach params after re-apply
    return EqWorldModel
