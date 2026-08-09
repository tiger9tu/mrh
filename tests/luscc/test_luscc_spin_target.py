import unittest

import numpy as np
from pyscf import gto, scf

from mrh.exploratory.luscc.solver import LUSCC
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom="H 0 0 0; H 1 0 0; H .2 1.6 .1; H 1.159166 1.3 -.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1))
        mo = las.localize_init_guess(((0, 1), (2, 3)), mf.mo_coeff)
        las.kernel(mo)
        cls.las = las

    def _run_target(self, smult):
        # This double excitation produces local-spin-impure fragment vectors,
        # forcing the numerical S^2 projector rather than CG spin coupling.
        luscc = LUSCC(
            self.las, [np.array((2, 0))], [np.array((2, 1))],
            opt=1, smult_si=smult)
        luscc.sisolver.nroots = 1
        energy, si = luscc.kernel()
        return luscc, energy, si

    def test_singlet_projector(self):
        luscc, energy, si = self._run_target(1)
        self.assertTrue(luscc.converged_si)
        self.assertEqual(energy.shape, (1,))
        self.assertAlmostEqual(si.s2[0], 0.0, 9)

    def test_triplet_projector(self):
        luscc, energy, si = self._run_target(3)
        self.assertTrue(luscc.converged_si)
        self.assertEqual(energy.shape, (1,))
        self.assertAlmostEqual(si.s2[0], 2.0, 9)


if __name__ == "__main__":
    unittest.main()
