import numpy as np
import pytest

from mrh.exploratory.luscc import LUSCC


def test_h4_luscc_energy(h4_las):
    # One fixed alpha charge-transfer generator and its de-excitation produce
    # a small, deterministic end-to-end LUSCC model space.
    energy, _ = LUSCC(
        h4_las,
        a_idxs=[np.array([2])],
        i_idxs=[np.array([0])],
    ).kernel()
    assert energy[0] == pytest.approx(-2.200190886022359, abs=1e-9)
    assert energy[0] <= h4_las.e_tot + 1e-10
