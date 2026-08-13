import numpy as np
from pyscf.fci.spin_op import contract_ss

from mrh.exploratory.luscc.spin import residual_gram
from mrh.my_pyscf.lassi import op_o0


class _SpinImpureProducts:
    nfrags = 2
    ncas_sub = (2, 2)

    def __init__(self):
        # Fixed-Ms fragment vectors deliberately mixing local singlet and
        # triplet components. No fragment spin quantum number is supplied.
        a = np.array([[0.0, 1.0], [2.0, 0.0]]) / np.sqrt(5.0)
        b = np.array([[0.0, 3.0], [-1.0, 0.0]]) / np.sqrt(10.0)
        c = np.array([[1.0, 0.0], [0.0, 2.0]]) / np.sqrt(5.0)
        d = np.array([[2.0, 0.0], [0.0, -1.0]]) / np.sqrt(5.0)
        self.ci = [[a, c], [b, d]]
        self._nelec = np.ones((2, 2, 2), dtype=int)

    def get_nelec_frs(self):
        return self._nelec


def test_residual_gram_matches_combined_fci_for_spin_impure_fragments():
    products = _SpinImpureProducts()
    k_factorized = residual_gram(products, spin=0)

    global_ci, nelec = op_o0.ci_outer_product(
        products.ci, products.ncas_sub, products.get_nelec_frs())
    qci = [contract_ss(ci, sum(products.ncas_sub), tuple(nel))
           for ci, nel in zip(global_ci, nelec)]
    k_fci = np.asarray([[np.vdot(left, right) for right in qci]
                        for left in qci])
    np.testing.assert_allclose(k_factorized, k_fci, atol=1e-12)
