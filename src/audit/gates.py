r"""Canonical C3 gate mechanics — THE single shared implementation (registered 2026-06-11).

Provenance: the pre-amendment ratio semantics (``None`` auto-pass / never-qualify) was planted
independently in TWO scripts (champion confirm + CPU batch) and caught twice in one day. Rule
going forward: gate mechanics live HERE and are imported, never copied.

Faithful semantics (C3-guar amendment, registered between champion r1/r2):

- For each $\epsilon$ cell, store BOTH boundaries $H_{\mathrm{cert}}, H_{\mathrm{meas}}$ where
  $H(\epsilon) = \max\{H \le h_{\max} : \mathrm{err}(H) \le \epsilon\}$ (0 if none).
- **Guarantee (one-sided):** pass iff $H_{\mathrm{cert}} \le H_{\mathrm{meas}}$ at ALL cells.
  $(0,0)$ PASSES — the certificate refuses a horizon the world also refuses.
  $H_{\mathrm{cert}} > H_{\mathrm{meas}}$ FAILS — that is anticonservatism, the thing C3 forbids.
- **Calibration (two-sided):** ratio $H_{\mathrm{cert}}/H_{\mathrm{meas}} \in$ band, evaluated
  ONLY on cells with both boundaries positive (a $(0,0)$ cell carries no calibration info).
- **Censoring caveat (registered):** a cell with both boundaries at $h_{\max}$ is
  pass-by-censoring; verdict language must scope claims to "within the audited horizon".
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_EPS_MULT = (2, 4, 8, 16)
DEFAULT_BAND = (0.5, 2.0)


def boundary_from_curve(curve: Sequence[float], eps: float) -> int:
    r"""Largest $H$ (1-indexed) with $\mathrm{curve}[H-1] \le \epsilon$; 0 if even $H{=}1$ fails.

    Replicates the historical ``experiments/p4_spine_stage1a.boundary_from_curve`` exactly
    (equivalence-tested in ``tests/test_p4_gates.py``); this copy is the canonical one for all
    NEW scripts.
    """
    h = 0
    for i, v in enumerate(curve):
        if v <= eps:
            h = i + 1
        else:
            break
    return h


def dual_boundary_cells(meas_q90: Sequence[float], cert_q90: Sequence[float],
                        delta_mean: float,
                        eps_mult: Sequence[int] = DEFAULT_EPS_MULT) -> list[dict]:
    r"""Per-$\epsilon$ cells with both boundaries + ratio (None unless both positive)."""
    cells = []
    for em in eps_mult:
        hm = boundary_from_curve(meas_q90, em * delta_mean)
        hc = boundary_from_curve(cert_q90, em * delta_mean)
        cells.append({"em": em, "h_meas": int(hm), "h_cert": int(hc),
                      "ratio": (hc / hm) if (hm and hc) else None})
    return cells


def faithful_guar(cells: Sequence[dict]) -> bool:
    r"""One-sided guarantee: $H_{\mathrm{cert}} \le H_{\mathrm{meas}}$ at every cell."""
    return bool(cells) and all(c["h_cert"] <= c["h_meas"] for c in cells)


def cal_band(cells: Sequence[dict], band: tuple[float, float] = DEFAULT_BAND) -> bool:
    r"""Two-sided calibration on both-positive cells only; False if no evaluable cell."""
    rs = [c["ratio"] for c in cells if c["ratio"] is not None]
    return bool(rs) and all(band[0] <= r <= band[1] for r in rs)


def violations(cells: Sequence[dict]) -> list[dict]:
    r"""Cells where the certificate is anticonservative (the C3-forbidden direction)."""
    return [c for c in cells if c["h_cert"] > c["h_meas"]]
