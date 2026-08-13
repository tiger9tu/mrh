"""Exact total-spin constraints for LAS product-state expansions."""

import numpy as np
from scipy import linalg
from pyscf.fci import addons as fci_addons
from pyscf.fci.spin_op import contract_ss

from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr


def _fermion_spin_shuffle(nelecs):
    """Phase from fragment-major to global spin-major operator ordering."""
    nperm = sum(sum(ne[0] for ne in nelecs[:frag]) * nelecs[frag][1]
                for frag in range(1, len(nelecs)))
    return -1.0 if nperm % 2 else 1.0


def _local_ladder(ci, norb, nelec, direction):
    """Apply a fragment S+ or S- operator to an FCI vector."""
    na, nb = map(int, nelec)
    out_nelec = (na + 1, nb - 1) if direction == "+" else (na - 1, nb + 1)
    if min(out_nelec) < 0 or max(out_nelec) > norb:
        return None, out_nelec
    out = None
    for orb in range(norb):
        if direction == "+":
            term = fci_addons.des_b(ci, norb, (na, nb), orb)
            term = fci_addons.cre_a(term, norb, (na, nb - 1), orb)
        else:
            term = fci_addons.des_a(ci, norb, (na, nb), orb)
            term = fci_addons.cre_b(term, norb, (na - 1, nb), orb)
        out = term if out is None else out + term
    if out is None or not np.any(out):
        return None, out_nelec
    return out, out_nelec


def _product_basis(las):
    """Expand LASSI root spaces into explicit lists of fragment factors."""
    lroots = get_lroots(las.ci)
    rootaddr, fragaddr = get_rootaddr_fragaddr(lroots)
    nelec_frs = las.get_nelec_frs()
    basis = []
    for state, root in enumerate(rootaddr):
        factors, nelecs = [], []
        for frag in range(las.nfrags):
            block = las.ci[frag][root]
            factors.append(np.asarray(block[fragaddr[frag, state]]
                                      if block.ndim > 2 else block))
            nelecs.append(tuple(map(int, nelec_frs[frag, root])))
        basis.append((factors, tuple(nelecs)))
    return basis


def _residual_terms(factors, nelecs, norb_f, target_s2):
    """Represent (S^2-target_s2)|product> as product-state terms."""
    terms = [(-target_s2, factors, nelecs)]
    source_shuffle = _fermion_spin_shuffle(nelecs)
    nfrag = len(factors)
    for a in range(nfrag):
        sfactors = list(factors)
        sfactors[a] = contract_ss(factors[a], norb_f[a], nelecs[a])
        terms.append((1.0, sfactors, nelecs))
    for a in range(nfrag):
        ma = (nelecs[a][0] - nelecs[a][1]) / 2.0
        for b in range(a + 1, nfrag):
            mb = (nelecs[b][0] - nelecs[b][1]) / 2.0
            terms.append((2.0 * ma * mb, factors, nelecs))
            for da, db in (("+", "-"), ("-", "+")):
                va, nea = _local_ladder(factors[a], norb_f[a], nelecs[a], da)
                vb, neb = _local_ladder(factors[b], norb_f[b], nelecs[b], db)
                if va is None or vb is None:
                    continue
                sfactors = list(factors)
                snelecs = list(nelecs)
                sfactors[a], sfactors[b] = va, vb
                snelecs[a], snelecs[b] = nea, neb
                snelecs = tuple(snelecs)
                # Local CI arrays use fragment-major spin ordering whereas
                # LASSI coefficients and the global FCI vector use spin-major
                # ordering. Spin ladders change this shuffle phase.
                phase = source_shuffle * _fermion_spin_shuffle(snelecs)
                terms.append((phase, sfactors, snelecs))
    return [(coef, fac, nel) for coef, fac, nel in terms if coef != 0]


def _product_overlap(left, right):
    lc, lf, ln = left
    rc, rf, rn = right
    if ln != rn:
        return 0.0
    value = np.conjugate(lc) * rc
    for bra, ket in zip(lf, rf):
        value *= np.vdot(bra, ket)
    return value


def residual_gram(las, spin):
    """Return K_ij=<Q psi_i|Q psi_j> without global FCI vectors."""
    if float(spin) < 0 or not np.isclose(2.0 * float(spin),
                                        round(2.0 * float(spin))):
        raise ValueError("target spin S must be a nonnegative integer or half-integer")
    target_s2 = float(spin) * (float(spin) + 1.0)
    basis = _product_basis(las)
    terms = [_residual_terms(*state, las.ncas_sub, target_s2)
             for state in basis]
    nstate = len(terms)
    dtype = np.result_type(*[factor.dtype for state in basis
                             for factor in state[0]])
    gram = np.zeros((nstate, nstate), dtype=dtype)
    for i in range(nstate):
        for j in range(i + 1):
            value = sum(_product_overlap(a, b)
                        for a in terms[i] for b in terms[j])
            gram[i, j] = value
            gram[j, i] = np.conjugate(value)
    return (gram + gram.conj().T) / 2


def solve_exact_spin(hamiltonian, overlap, residual, spin, lin_dep_tol=1e-10,
                     spin_tol=1e-10):
    """Solve H in the metric-orthonormal numerical null space of K."""
    mval, mvec = linalg.eigh(overlap)
    mcut = max(lin_dep_tol, lin_dep_tol * max(1.0, float(mval[-1])))
    keep = mval > mcut
    if not np.any(keep):
        raise ValueError("LUSCC product basis is numerically linearly dependent")
    xmat = mvec[:, keep] / np.sqrt(mval[keep])
    korth = xmat.conj().T @ residual @ xmat
    kval, kvec = linalg.eigh((korth + korth.conj().T) / 2)
    # Eigenvalues of K are squared residual norms. ``spin_tol`` therefore
    # applies to their square roots.
    kscale = max(1.0, float(np.max(np.abs(kval))))
    kcut = max(spin_tol**2 * kscale,
               np.finfo(korth.real.dtype).eps * len(kval) * kscale * 10.0)
    null = np.abs(kval) <= kcut
    if not np.any(null):
        raise ValueError(
            f"No exact total-spin S={spin:g} vector exists in the LUSCC "
            f"space within tolerance {spin_tol:g}")
    projector = xmat @ kvec[:, null]
    hcon = projector.conj().T @ hamiltonian @ projector
    energy, vec = linalg.eigh((hcon + hcon.conj().T) / 2)
    coeff = projector @ vec
    residual_norm = np.sqrt(np.maximum(0.0, np.real(np.einsum(
        "ip,ij,jp->p", coeff.conj(), residual, coeff))))
    return energy, coeff, residual_norm, kval


def exact_spin_basis(overlap, residual, spin, lin_dep_tol=1e-10,
                     spin_tol=1e-10):
    """Return a raw-coefficient orthonormal basis for the exact-spin space."""
    mval, mvec = linalg.eigh((overlap + overlap.conj().T) / 2)
    mcut = max(lin_dep_tol, lin_dep_tol * max(1.0, float(mval[-1])))
    keep = mval > mcut
    if not np.any(keep):
        raise ValueError("LUSCC product basis is numerically linearly dependent")
    xmat = mvec[:, keep] / np.sqrt(mval[keep])
    korth = xmat.conj().T @ residual @ xmat
    kval, kvec = linalg.eigh((korth + korth.conj().T) / 2)
    kscale = max(1.0, float(np.max(np.abs(kval))))
    kcut = max(spin_tol**2 * kscale,
               np.finfo(korth.real.dtype).eps * len(kval) * kscale * 10.0)
    null = np.abs(kval) <= kcut
    if not np.any(null):
        raise ValueError(
            f"No exact total-spin S={spin:g} vector exists in the LUSCC "
            f"space within tolerance {spin_tol:g}")
    return xmat @ kvec[:, null], kval
