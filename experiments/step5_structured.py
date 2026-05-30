r"""Step 5: structured-state observation wrapper + the coordinate-path payoff.

Shows three things on **real PushT state**:

  1. ``extract_pusht_vectors`` turns the env's SE(2) state into 6 type-1 vectors.
  2. The :class:`StructuredStateEncoder` is exactly SO(2)-equivariant for
     *arbitrary* continuous angles (down to the float floor), so the JEPA cost is
     rotation-invariant for any rotation — not just multiples of $90^\circ$.
  3. Head-to-head with the pixel :class:`SteerableEncoder` on the same frames:
     the pixel path is exact at $90^\circ$ but hits the bilinear-interpolation
     floor at $45^\circ$; the coordinate path is exact at both.

This is the empirical case for moving observations onto coordinates (the bridge
to the e3nn SE(3) encoder in Step 6).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step5_structured.py
"""

import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torchvision.transforms.functional import rotate as tv_rotate  # noqa: E402

import stable_worldmodel as swm  # noqa: E402

from src.geometry.so2 import rotate_vectors  # noqa: E402
from src.models.structured import StructuredStateEncoder, extract_pusht_vectors  # noqa: E402
from src.models.eqjepa import SteerableEncoder  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    size = 65

    banner("[A] Extract structured SE(2) state from real PushT observations")
    world = swm.World("swm/PushT-v1", num_envs=8, image_shape=(size, size))
    world.reset(seed=0)
    X = extract_pusht_vectors(world.infos)  # (B, 6, 2)
    pixels = torch.as_tensor(
        world.infos["pixels"][:, -1]
    ).permute(0, 3, 1, 2).float().div(255.0)  # (B, 3, H, W)
    world.close()
    names = ["agent_pos", "block_pos", "goal_pos", "block_dir", "goal_dir", "agent_vel"]
    print(f"    structured state X: {tuple(X.shape)}  (6 type-1 vectors)")
    for i, nm in enumerate(names):
        print(f"      {nm:10s} env0 = ({X[0, i, 0]:+.3f}, {X[0, i, 1]:+.3f})")
    print(f"    pixels for the side-by-side: {tuple(pixels.shape)}")

    banner("[B] Coordinate encoder — equivariance on real state for ANY angle")
    enc = StructuredStateEncoder(n_vec=6, latent_dim=128, hidden=64).eval()
    z = enc.encode_vectors(X)
    angles = [37.0, 45.0, 90.0, 123.4, 211.7]
    worst = 0.0
    print("    angle    | max|E(R.x)-R.E(x)| | cost drift")
    print("    ---------+--------------------+-----------")
    for deg in angles:
        a = math.radians(deg)
        left = enc.encode_vectors(rotate_vectors(X, a))
        right = rotate_vectors(z, a)
        eq = (left - right).abs().max().item()
        base = (enc(X[0:4]) - enc(X[4:8])).norm()
        rot = (enc(rotate_vectors(X[0:4], a)) - enc(rotate_vectors(X[4:8], a))).norm()
        drift = (rot - base).abs().item()
        worst = max(worst, eq, drift)
        print(f"    {deg:6.1f}d  |     {eq:.2e}     |  {drift:.2e}")
    print(f"    worst (coordinate path, all angles) = {worst:.2e}")

    banner("[C] Pixel SteerableEncoder on the SAME frames — 90 vs 45 deg")
    penc = SteerableEncoder(in_channels=3, latent_dim=128, n_rot=4).eval()

    def pixel_cost(imgs):
        z = penc(imgs)
        return (z[0:4] - z[4:8]).norm().item()

    base_px = pixel_cost(pixels)
    rot90 = torch.rot90(pixels, 1, dims=(-2, -1))  # exact 90 deg pixel permutation
    rot45 = tv_rotate(pixels, 45.0)  # 45 deg -> bilinear interpolation
    drift90 = abs(pixel_cost(rot90) - base_px)
    drift45 = abs(pixel_cost(rot45) - base_px)
    print(f"    pixel cost drift @ 90 deg (rot90, exact)      = {drift90:.2e}")
    print(f"    pixel cost drift @ 45 deg (interpolated)      = {drift45:.2e}")

    banner("STEP 5 SUMMARY")
    print(f"    coordinate path : exact for all angles (worst {worst:.1e})")
    print(f"    pixel path      : exact @90deg ({drift90:.1e}) but floors @45deg ({drift45:.1e})")
    ratio = drift45 / max(worst, 1e-12)
    print(f"    => at 45 deg the coordinate path is ~{ratio:.0e}x more rotation-stable")
    assert worst < 1e-4, f"coordinate equivariance broke: {worst:.2e}"
    print("    PASS: structured-state encoder removes the pixel interpolation floor.")


if __name__ == "__main__":
    main()
