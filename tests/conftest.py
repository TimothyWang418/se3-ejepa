r"""Shared pytest fixtures.

Test isolation for the global default dtype. A few experiments (Step 50/51/52) work in ``float64`` so that
"exact / invariant" residuals are unambiguous, while the rest of the codebase (Step 13/24/43, the SE(3)/SO(2)
models) uses the PyTorch default ``float32``. ``torch.set_default_dtype`` is **global and persistent**, so without
a reset a float64 test would leave float64 active for whatever float32 test runs next (cross-test contamination,
e.g. ``float != double`` in ``rotate_points``). This autouse fixture pins the default to ``float32`` around every
test; the float64 tests opt in explicitly at the top of their own functions.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)
