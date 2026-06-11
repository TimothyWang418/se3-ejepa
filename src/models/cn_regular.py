r"""$C_N$-equivariant latent predictor on regular-representation fibers (paper3 / P4-step1).

The pixel path's encoder (:class:`~src.models.eqjepa.SteerableEncoder`) outputs a latent
$z \in \mathbb{R}^{F \cdot N}$ organised as $F$ copies of the **regular representation** of the
cyclic group $C_N$: viewing $z$ as $(F, N)$, a rotation $g \in C_N$ acts by the cyclic shift

$$ (\rho(g) z)[f, n] \;=\; z[f,\, (n - g) \bmod N], $$

which is orthogonal (a permutation). Theorem A (paper2) needs **both** $E$ and the predictor $f$
equivariant; the repo's :class:`~src.models.eqjepa.LatentPredictor` is a plain MLP (no structure),
and the VN predictors (:mod:`src.models.structured`) implement $\mathrm{SO}(3)$ acting on 3-vector
channels — a *different* group action. This module supplies the missing piece.

**Equivariant linear map on regular fibers = group convolution** (Cohen & Welling 2016, *Group
Equivariant Convolutional Networks*): the most general linear map commuting with all shifts is

$$ h[f', n] \;=\; \sum_{f=1}^{F} \sum_{k=0}^{N-1} W[f', f, k]\; z[f,\, (n - k) \bmod N], $$

i.e. per field-pair a circulant matrix; implemented below as a 1-D convolution with circular
padding along the group axis. A **pointwise** nonlinearity applied per group element commutes with
the permutation action, hence is equivariant (the standard regular-field construction).

**Action conditioning.** The PushT action is a 2-D target position; centred on the canvas it
transforms by the standard (frequency-1) representation, $a \mapsto R(\theta_g) a$ with
$\theta_g = 2\pi g/N$. The frequency-1 **lift** to a function on the group,

$$ A_a[n] \;=\; a_x \cos(2\pi n/N) + a_y \sin(2\pi n/N), $$

intertwines the two actions: replacing $a$ by $R(\theta_g)a$ gives, by the angle-sum identities,
$A_{R_g a}[n] = A_a[(n - g) \bmod N]$ — exactly the regular shift. (Sign conventions — shift
direction vs. rotation orientation vs. e2cnn's fiber permutation — are *fixed by the unit tests*
in ``tests/test_p4_step1.py``, which check the predictor against the encoder's own
``GeometricTensor.transform``; do not change one without the other.) A second, invariant channel
carries $\lVert a \rVert$ as a constant-over-$n$ field (constants are shift-invariant).

The predictor mirrors :class:`LatentPredictor`'s residual structure ($z + \Delta$, the standard
stabiliser for learned latent rollouts), and matches its ``forward(z, a) -> z'`` interface so it
drops into :class:`~src.models.eqjepa.EqJEPA` unchanged.

**Honest scope (registered):** equivariance here is *exact in the fiber* (float precision — unlike
the encoder, no pixel interpolation is involved). The PushT *environment*'s $\mathrm{SO}(2)$
symmetry is broken by the square workspace boundary (only $C_4$ survives exactly); wall-contact
transitions therefore contribute to the *measured* $\hat\epsilon_{\max}$ — on-thesis (Theorem B
absorbs measured residuals; nothing is assumed exact).

forward: ``(B, F*N), (B, 2) -> (B, F*N)``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GroupConv1x1(nn.Module):
    r"""$C_N$-equivariant linear layer on regular fibers: per-field-pair circulant mixing.

    ``(B, F_in, N) -> (B, F_out, N)`` via conv1d with circular padding (kernel covers all $N$
    group offsets). Bias is per-output-field, constant over the group axis (equivariant).
    """

    def __init__(self, f_in: int, f_out: int, n_rot: int):
        super().__init__()
        self.n_rot = n_rot
        # weight[f_out, f_in, k] — the circulant kernel over group offsets k.
        self.conv = nn.Conv1d(f_in, f_out, kernel_size=n_rot, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # circular padding on the group axis, then valid conv => circulant action.
        xp = F.pad(x, (0, self.n_rot - 1), mode="circular")
        return self.conv(xp)


class CNRegularPredictor(nn.Module):
    r"""Residual $C_N$-equivariant action-conditioned predictor on regular fibers.

    Args:
        latent_dim: $F \cdot N$ (must be divisible by ``n_rot``).
        action_dim: must be 2 (planar position action; frequency-1 lift).
        n_rot: $N$, the cyclic order (must match the encoder's ``n_rot``).
        hidden_fields: $H$, number of hidden regular fields.
        action_center: subtracted before the lift. The swm PushT env default is
            ``relative=True`` — actions are **displacement vectors** in $[-1,1]^2$, which rotate
            about the **origin** as free vectors ⇒ default centre $(0,0)$, scale $1$. For
            absolute-target mode (``relative=False``) pass the canvas centre and a pixel scale.
        action_scale: scale applied after centring (keeps the lift O(1)).

    forward: ``(B, F*N), (B, 2) -> (B, F*N)`` with
    $f(\rho(g) z,\, R(\theta_g) a) = \rho(g)\, f(z, a)$ exactly (float precision).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        n_rot: int = 16,
        hidden_fields: int = 32,
        action_center: tuple[float, float] = (0.0, 0.0),
        action_scale: float = 1.0,
    ):
        super().__init__()
        if action_dim != 2:
            raise ValueError("CNRegularPredictor's frequency-1 action lift requires action_dim=2.")
        if latent_dim % n_rot != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by n_rot ({n_rot}).")
        self.latent_dim = latent_dim
        self.n_rot = n_rot
        self.n_fields = latent_dim // n_rot
        self.register_buffer("a_center", torch.tensor(action_center, dtype=torch.float32))
        self.a_scale = float(action_scale)
        # phases of the group elements: 2*pi*n/N, n = 0..N-1
        phase = 2.0 * torch.pi * torch.arange(n_rot, dtype=torch.float32) / n_rot
        self.register_buffer("cos_n", torch.cos(phase))
        self.register_buffer("sin_n", torch.sin(phase))

        f_in = self.n_fields + 2  # latent fields + [freq-1 action lift, invariant |a|]
        self.net = nn.Sequential(
            GroupConv1x1(f_in, hidden_fields, n_rot),
            nn.ReLU(inplace=True),
            GroupConv1x1(hidden_fields, hidden_fields, n_rot),
            nn.ReLU(inplace=True),
            GroupConv1x1(hidden_fields, self.n_fields, n_rot),
        )

    def lift_action(self, a: torch.Tensor) -> torch.Tensor:
        r"""Frequency-1 lift + invariant norm channel: ``(B, 2) -> (B, 2, N)``.

        $A_a[n] = \tilde a_x \cos(2\pi n/N) + \tilde a_y \sin(2\pi n/N)$ with
        $\tilde a = (a - c)/s$; second channel $= \lVert\tilde a\rVert$ (constant over $n$).
        """
        at = (a - self.a_center) / self.a_scale  # (B, 2), centred + scaled
        freq1 = at[:, :1] * self.cos_n + at[:, 1:2] * self.sin_n  # (B, N)
        inv = at.norm(dim=1, keepdim=True).expand(-1, self.n_rot)  # (B, N)
        return torch.stack([freq1, inv], dim=1)  # (B, 2, N)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        zf = z.view(B, self.n_fields, self.n_rot)  # (B, F, N)
        x = torch.cat([zf, self.lift_action(a)], dim=1)  # (B, F+2, N)
        dz = self.net(x).reshape(B, self.latent_dim)
        return z + dz  # residual, like LatentPredictor
