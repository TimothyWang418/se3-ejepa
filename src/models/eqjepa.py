r"""EqJEPA — a JEPA-style latent world model for the ``stable-worldmodel`` harness.

This is the **Step 2 skeleton**: an *ordinary* (non-equivariant) encoder plus a
latent dynamics predictor, wired to satisfy SWM's ``Costable`` protocol so it can
be driven by the CEM / iCEM / MPPI planning solvers and ``World.evaluate``.

Mathematical sketch
--------------------
We learn a latent world model in the spirit of JEPA (LeCun 2022; Bardes et al.
V-JEPA 2024) rather than a pixel-space generative model:

* **Encoder** $E_\theta : \mathcal{X} \to \mathbb{R}^D$ maps an observation
  (here: a rendered image) to a latent $z = E_\theta(x)$.
* **Latent predictor** $f_\phi : \mathbb{R}^D \times \mathbb{R}^A \to \mathbb{R}^D$
  advances the latent under an action, $\hat z_{k+1} = f_\phi(\hat z_k, a_k)$.
* **Planning cost**. Given a goal observation $x_g$ with latent
  $z_g = E_\theta(x_g)$ and a candidate action sequence $a_{0:H-1}$, we roll the
  latent forward and score the terminal mismatch in latent space:

  $$ \hat z_0 = E_\theta(x_0), \quad \hat z_{k+1} = f_\phi(\hat z_k, a_k),
     \qquad
     \mathcal{C}(a_{0:H-1}) = \lVert \hat z_H - z_g \rVert_2^2 . $$

  The solver minimises $\mathcal{C}$ over candidate action sequences (CEM).

The whole point of this project is that $E_\theta$ should eventually be
**SE(3)/SO(2)-equivariant** and $f_\phi$ **jointly equivariant** in $(\hat z, a)$,
so that $f_\phi(\rho(g)\hat z, \rho(g) a) = \rho(g) f_\phi(\hat z, a)$. Step 2 keeps
both ordinary so we have a known-good control before changing the geometry.

Contract with the SWM solvers (see ``stable_worldmodel/solver/cem.py``):
    ``get_cost(info_dict, candidates) -> Tensor`` of shape ``(B, S)`` exactly,
    where ``B`` is the number of envs in the current batch and ``S`` the number
    of action candidates. ``info_dict`` values arrive pre-expanded to
    ``(B, S, *per_env_shape)``; image keys (``pixels``/``goal``) arrive as
    ``(B, S, T, C, H, W)`` float tensors (channel-first, already transformed by
    the policy's ``_prepare_info``).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ConvEncoder", "SteerableEncoder", "LatentPredictor", "EqJEPA"]


class ConvEncoder(nn.Module):
    r"""Ordinary (non-equivariant) CNN encoder $E_\theta : \mathbb{R}^{C\times H\times W} \to \mathbb{R}^D$.

    Step 3 will replace this with an ``escnn`` SO(2)-steerable CNN; the rest of
    :class:`EqJEPA` is written so that swap is local to this class.

    forward: ``(N, C, H, W) -> (N, latent_dim)``
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, width: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),  # H/2
            nn.GroupNorm(8, width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),  # H/4
            nn.GroupNorm(8, 2 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1),  # H/8
            nn.GroupNorm(8, 4 * width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (N, 4*width, 1, 1)
            nn.Flatten(),  # (N, 4*width)
        )
        self.head = nn.Linear(4 * width, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, latent_dim)
        return self.head(self.net(x))


class SteerableEncoder(nn.Module):
    r"""$C_N$-steerable (SO(2)) encoder $E_\theta$ with an *equivariant* latent.

    Uses ``e2cnn`` (escnn's E(2) predecessor — escnn's 3D path needs ``lie-learn``,
    which has no arm64 wheels). The RGB image is treated as 3 scalar (trivial)
    fields; hidden layers carry the regular representation of $C_N$; a final
    conv maps to ``latent_dim / N`` regular fields, and a *global spatial average
    pool* (rotation-equivariant) yields the latent.

    The latent then satisfies $E_\theta(g\cdot x) = \rho(g)\, E_\theta(x)$ with
    $\rho$ the **orthogonal** regular representation of $C_N$ — orthogonality is
    what makes the JEPA cost $\lVert\hat z_H - z_g\rVert_2^2$ rotation-invariant.

    Caveat (verified empirically, see ``tests/test_e2cnn_equivariance.py``): the
    steerable conv is exactly $C_N$-equivariant, but *measuring* it via pixel-grid
    rotation is only float-exact for $90^\circ$-multiples; non-grid angles incur
    bilinear-interpolation error. Exact continuous SO(2)/SO(3) needs the
    coordinate/point-cloud path (Steps 5–6).

    forward: ``(N, C, H, W) -> (N, latent_dim)``
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        n_rot: int = 4,
        width: int = 8,
    ):
        super().__init__()
        from e2cnn import gspaces
        from e2cnn import nn as enn

        if latent_dim % n_rot != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by n_rot ({n_rot}) "
                "so the latent decomposes into whole regular fields."
            )
        self.latent_dim = latent_dim
        self.n_rot = n_rot
        self.r2 = gspaces.Rot2dOnR2(N=n_rot)

        self.in_type = enn.FieldType(self.r2, in_channels * [self.r2.trivial_repr])
        h1 = enn.FieldType(self.r2, width * [self.r2.regular_repr])
        h2 = enn.FieldType(self.r2, (2 * width) * [self.r2.regular_repr])
        h3 = enn.FieldType(self.r2, (4 * width) * [self.r2.regular_repr])
        n_fields = latent_dim // n_rot
        self.out_type = enn.FieldType(self.r2, n_fields * [self.r2.regular_repr])

        self.net = enn.SequentialModule(
            enn.R2Conv(self.in_type, h1, kernel_size=5, stride=2, padding=2, bias=False),
            enn.InnerBatchNorm(h1),
            enn.ReLU(h1),
            enn.R2Conv(h1, h2, kernel_size=5, stride=2, padding=2, bias=False),
            enn.InnerBatchNorm(h2),
            enn.ReLU(h2),
            enn.R2Conv(h2, h3, kernel_size=5, stride=2, padding=2, bias=False),
            enn.InnerBatchNorm(h3),
            enn.ReLU(h3),
            enn.R2Conv(h3, self.out_type, kernel_size=3, padding=1, bias=False),
        )
        self.pool = enn.PointwiseAdaptiveAvgPool(self.out_type, 1)

    def encode_geometric(self, x: torch.Tensor):
        r"""Return the pooled ``GeometricTensor`` (shape ``(N, latent_dim, 1, 1)``).

        Exposed so the equivariance unit test can use ``.transform(g)`` on the
        output fiber directly.
        """
        from e2cnn import nn as enn

        gx = enn.GeometricTensor(x, self.in_type)
        return self.pool(self.net(gx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, latent_dim)
        return self.encode_geometric(x).tensor.flatten(1)


class LatentPredictor(nn.Module):
    r"""Residual latent dynamics $f_\phi(\hat z, a) = \hat z + g_\phi([\hat z, a])$.

    Predicting a *residual* keeps the map close to identity at initialisation, a
    standard stabiliser for learned latent rollouts (cf. DINO-WM / PLDM dynamics
    heads). Step 6 makes this jointly equivariant in $(\hat z, a)$.

    forward: ``((N, latent_dim), (N, action_dim)) -> (N, latent_dim)``
    """

    def __init__(self, latent_dim: int = 128, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # z: (N, latent_dim), a: (N, action_dim) -> (N, latent_dim)
        return z + self.net(torch.cat([z, a], dim=-1))


class EqJEPA(nn.Module):
    r"""Latent world model implementing SWM's ``Costable`` protocol.

    Args:
        in_channels: image channels (3 for RGB).
        latent_dim: dimension $D$ of the latent $z$.
        action_dim: dimension $A$ of a single action.
        encoder: optional pre-built encoder module (defaults to :class:`ConvEncoder`).
            Pass a steerable/equivariant encoder here in Step 3+.
        predictor: optional pre-built predictor (defaults to :class:`LatentPredictor`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        action_dim: int = 2,
        encoder: nn.Module | None = None,
        predictor: nn.Module | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.encoder = encoder or ConvEncoder(in_channels, latent_dim)
        self.predictor = predictor or LatentPredictor(latent_dim, action_dim)
        # SWM's eval_wm.py sets this flag on the model; harmless to expose.
        self.interpolate_pos_encoding = False

    # -- core latent ops -----------------------------------------------------

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        r"""Encode an image (or batch of images) to latents.

        Accepts a trailing ``(C, H, W)`` with arbitrary leading dims and returns
        latents with the same leading dims.

        ``(*lead, C, H, W) -> (*lead, latent_dim)``
        """
        lead = pixels.shape[:-3]
        chw = pixels.shape[-3:]
        z = self.encoder(pixels.reshape(-1, *chw))  # (prod(lead), D)
        return z.reshape(*lead, self.latent_dim)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        r"""Roll the latent forward under an action sequence.

        $\hat z_{k+1} = f_\phi(\hat z_k, a_k)$ for $k = 0,\dots,H-1$.

        ``z0: (N, D), actions: (N, H, A) -> z_H: (N, D)``
        """
        z = z0
        horizon = actions.shape[1]
        for k in range(horizon):
            z = self.predictor(z, actions[:, k])
        return z

    # -- Costable protocol ----------------------------------------------------

    @staticmethod
    def _last_frame_chw(img: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        r"""Normalise an image info tensor to ``(B, S, C, H, W)`` (last time frame).

        Handles both ``(B, S, T, C, H, W)`` (the solver-expanded layout) and an
        already-squeezed ``(B, S, C, H, W)``. Returns the tensor plus ``(B, S)``.
        """
        if img.ndim == 6:  # (B, S, T, C, H, W) -> take last time frame
            img = img[:, :, -1]
        elif img.ndim != 5:
            raise ValueError(
                f"Expected image tensor of ndim 5 or 6, got shape {tuple(img.shape)}"
            )
        b, s = img.shape[0], img.shape[1]
        return img, b, s

    def criterion(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        r"""Terminal latent-space cost $\lVert \hat z_H - z_g \rVert_2^2$.

        ``action_candidates: (B, S, H, A) -> cost: (B, S)``
        """
        pixels, b, s = self._last_frame_chw(info_dict["pixels"])  # (B,S,C,H,W)
        goal, _, _ = self._last_frame_chw(info_dict["goal"])  # (B,S,C,H,W)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        pixels = pixels.to(device=device, dtype=dtype)
        goal = goal.to(device=device, dtype=dtype)
        actions = action_candidates.to(device=device, dtype=dtype)

        c, h, w = pixels.shape[-3:]
        z0 = self.encoder(pixels.reshape(b * s, c, h, w))  # (B*S, D)
        zg = self.encoder(goal.reshape(b * s, c, h, w))  # (B*S, D)

        acts = actions.reshape(b * s, actions.shape[2], actions.shape[3])  # (B*S,H,A)
        zH = self.rollout(z0, acts)  # (B*S, D)

        cost = F.mse_loss(zH, zg, reduction="none").mean(dim=-1)  # (B*S,)
        return cost.reshape(b, s)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        r"""Entry point the solvers call. See :meth:`criterion`.

        ``action_candidates: (B, S, H, A) -> cost: (B, S)``
        """
        assert "pixels" in info_dict, "info_dict must contain 'pixels'"
        assert "goal" in info_dict, "info_dict must contain 'goal'"
        return self.criterion(info_dict, action_candidates)
