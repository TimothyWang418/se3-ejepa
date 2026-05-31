r"""Isotypic-block bookkeeping for $G$-adapted latent layouts.

A latent that carries an orthogonal representation $\rho:G\to O(n)$ decomposes
into **isotypic components** $V_i^{\oplus m_i}$ — $m_i$ copies (the *multiplicity*)
of an irreducible of dimension $d_i$. **Proposition 1** (Schur block-isotropy, see
``papers/equivariant_lejepa.md`` §3) says a $G$-invariant second moment is then
block-diagonal, $\Sigma=\bigoplus_i\mathbf I_{d_i}\otimes B_i$, with one free
$m_i\times m_i$ Gram block $B_i$ per isotypic component.

:class:`IrrepBlock` is the passive descriptor of *where* each such block lives in
the flattened latent. It is intentionally model- and training-agnostic so both the
equivariant encoder (:mod:`src.models.se3`, which *produces* the layout) and the
regularisers (:mod:`src.training.sigreg`, which *consume* it) can share one type —
hence its home in :mod:`src.geometry` rather than either caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["IrrepBlock"]


@dataclass(frozen=True)
class IrrepBlock:
    r"""One isotypic block of a $G$-adapted latent layout.

    The flattened latent ``z`` (shape ``(N, n)``) is partitioned into contiguous
    blocks; block $i$ occupies columns ``[start, start + d * m)`` and is read as
    ``m`` copies (multiplicity) of an irrep of dimension ``d``. For an SO(3)
    Vector-Neuron latent: scalars are ``d=1`` and vectors are ``d=3``.

    Attributes:
        name: human label (e.g. ``"0e"`` scalars, ``"1o"`` vectors).
        start: first column index of the block in the flattened latent.
        d: irrep dimension $d_i$ (1 for ``0e``, 3 for ``1o``).
        m: multiplicity $m_i$ (number of copies).
    """

    name: str
    start: int
    d: int
    m: int

    @property
    def width(self) -> int:
        r"""Number of latent columns this block spans, $d_i m_i$."""
        return self.d * self.m

    @property
    def stop(self) -> int:
        return self.start + self.width

    def slice(self, z: torch.Tensor) -> torch.Tensor:
        r"""Extract this block from a flattened latent. ``(N, n) -> (N, d*m)``."""
        return z[:, self.start : self.stop]
