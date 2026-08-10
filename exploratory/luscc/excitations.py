import numpy as np
from itertools import combinations, combinations_with_replacement
from pyscf.fci import addons as fci_addons
from pyscf.fci import cistring


def _spin_cases(spatial_indices, norb):
    """Return unique spin-orbital realizations and their beta counts."""
    cases = [(np.asarray(spatial_indices, dtype=np.int64), 0)]
    for position in range(len(spatial_indices)):
        cases += [
            (np.asarray(indices) + np.eye(1, len(indices), position,
                                         dtype=np.int64).ravel() * norb,
             nbeta + 1)
            for indices, nbeta in cases.copy()
        ]
    unique = {}
    for indices, nbeta in cases:
        key = tuple(sorted(indices.tolist()))
        unique.setdefault(key, (indices, nbeta))
    return list(unique.values())


def generate_interfragment_excitations(norb, ncas_sub, spin_adapted_singles=True):
    """Enumerate the spin-orbital singles and doubles used by LAS-LUSCC.

    This is the excitation-index portion formerly obtained by constructing a
    LAS-UCC operator.  Keeping it here removes any dependency on the abandoned
    LAS-USCC implementation.
    """
    offsets = np.append(0, np.cumsum(ncas_sub))
    singles = np.zeros((norb, norb), dtype=bool)
    for source, target in combinations(range(len(ncas_sub)), 2):
        singles[offsets[target]:offsets[target + 1],
                offsets[source]:offsets[source + 1]] = True
    creation_singles, annihilation_singles = np.where(singles)
    spatial_creations = [(int(index),) for index in creation_singles]
    spatial_annihilations = [(int(index),) for index in annihilation_singles]

    pairs = list(zip(*np.tril_indices(norb)))
    fragment = np.repeat(np.arange(len(ncas_sub)), ncas_sub)
    for creation, annihilation in combinations_with_replacement(pairs, 2):
        if np.unique(fragment[np.asarray(creation + annihilation)]).size > 1:
            spatial_creations.append(creation)
            spatial_annihilations.append(annihilation)

    a_idxs = []
    i_idxs = []
    for creation, annihilation in zip(spatial_creations,
                                      spatial_annihilations):
        if len(creation) == 1:
            spins = (0, 1) if spin_adapted_singles else (0, 1)
            for spin in spins:
                a_idxs.append(np.asarray(creation, dtype=np.int64) + spin*norb)
                i_idxs.append(np.asarray(annihilation, dtype=np.int64) + spin*norb)
            continue

        creation_cases = _spin_cases(creation, norb)
        annihilation_cases = _spin_cases(annihilation, norb)
        for creation_case, (creation_so, creation_beta) in enumerate(
                creation_cases):
            if np.unique(creation_so).size != len(creation_so):
                continue
            for annihilation_case, (annihilation_so, annihilation_beta) in enumerate(
                    annihilation_cases):
                if creation_beta != annihilation_beta:
                    continue
                if np.array_equal(creation_so, annihilation_so):
                    continue
                if (creation == annihilation
                        and creation_case > annihilation_case):
                    continue
                if np.unique(annihilation_so).size != len(annihilation_so):
                    continue
                a_idxs.append(creation_so)
                i_idxs.append(annihilation_so)
    return a_idxs, i_idxs


def _as_ci_matrix(ci0, norb, neleca, nelecb):
    """
    Ensure CI is a 2D array with shape (na, nb) where
      na = C(norb, neleca), nb = C(norb, nelecb).

    Accepts:
      - already-shaped (na, nb)
      - flat (na*nb,) vector in PySCF's row-major convention.
    """
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = np.asarray(ci0)
    if ci0.ndim == 2:
        if ci0.shape != (na, nb):
            raise ValueError(f"CI matrix has shape {ci0.shape}, expected {(na, nb)}")
        return ci0
    if ci0.ndim == 1:
        if ci0.size != na * nb:
            raise ValueError(f"Flat CI has size {ci0.size}, expected {na*nb}")
        return ci0.reshape((na, nb))
    raise ValueError(f"CI must be 1D or 2D, got ndim={ci0.ndim}")


def apply_operator_string_fci(ci0, norb, neleca_nelecb, ops):
    """
    Efficiently apply an *ordered* string of creation/annihilation operators to an FCI CI array,
    using PySCF's built-in FCI addons (des_a/cre_a/des_b/cre_b).

    Parameters
    ----------
    ci0 : (na, nb) array or flat array
        CI coefficients. Rows=alpha strings, cols=beta strings.
    norb : int
        Number of spatial orbitals.
    neleca_nelecb : (int, int)
        (neleca, nelecb) for ci0.
    ops : list
        Operator string in the order it acts on |psi>.
        Each op can be given as a dict or tuple:

        Tuple forms:
          (kind, orb, spin)
            kind: 'ann' or 'cre'   (also accepts 'a','c','des','create', etc via normalization below)
            orb : int (0-based spatial orbital)
            spin: 'a'/'alpha'/0 or 'b'/'beta'/1

        Dict form:
          {'kind': 'ann'/'cre', 'orb': p, 'spin': 'alpha'/'beta'}

    Returns
    -------
    ci : 2D np.ndarray
        New CI coefficients after applying ops.
    (neleca, nelecb) : (int, int)
        Updated electron counts.
    """
    neleca, nelecb = neleca_nelecb
    ci = _as_ci_matrix(ci0, norb, neleca, nelecb)

    def norm_kind(k):
        k = str(k).lower()
        if k in ('ann', 'a', 'des', 'destroy', 'annihilate', 'annihilation'):
            return 'ann'
        if k in ('cre', 'c', 'create', 'creation'):
            return 'cre'
        raise ValueError(f"Unknown operator kind: {k!r}")

    def norm_spin(s):
        if s in (0, '0'):
            return 'alpha'
        if s in (1, '1'):
            return 'beta'
        s = str(s).lower()
        if s in ('a', 'alpha', 'α', 'up'):
            return 'alpha'
        if s in ('b', 'beta', 'β', 'down'):
            return 'beta'
        raise ValueError(f"Unknown spin label: {s!r}")

    for op in ops:
        if isinstance(op, dict):
            kind = norm_kind(op['kind'])
            orb = int(op['orb'])
            spin = norm_spin(op['spin'])
        else:
            kind = norm_kind(op[0])
            orb = int(op[1])
            spin = norm_spin(op[2])

        if not (0 <= orb < norb):
            raise ValueError(f"orbital index out of range: {orb} (norb={norb})")

        # Optional strict checks (uncomment if you prefer raising instead of returning zeros)
        # if kind == 'ann':
        #     if spin == 'alpha' and neleca <= 0:
        #         raise ValueError("Cannot annihilate alpha electron: neleca=0")
        #     if spin == 'beta' and nelecb <= 0:
        #         raise ValueError("Cannot annihilate beta electron: nelecb=0")

        if kind == 'ann':
            if spin == 'alpha':
                if neleca <= 0:
                    return None, (neleca, nelecb)  # annihilating from empty state gives zero
                ci = fci_addons.des_a(ci, norb, (neleca, nelecb), orb)
                neleca -= 1
            else:
                if nelecb <= 0:
                    return None, (neleca, nelecb)  # annihilating from empty state gives zero
                ci = fci_addons.des_b(ci, norb, (neleca, nelecb), orb)
                nelecb -= 1

        else:  # kind == 'cre'
            if spin == 'alpha':
                if neleca >= norb:
                    return None, (neleca, nelecb)  # creating beyond full occupation gives zero
                ci = fci_addons.cre_a(ci, norb, (neleca, nelecb), orb)
                neleca += 1

            else:
                if nelecb >= norb:
                    return None, (neleca, nelecb)  # creating beyond full occupation gives zero
                ci = fci_addons.cre_b(ci, norb, (neleca, nelecb), orb)
                nelecb += 1

        # Check if CI is effectively zero (all elements below threshold)
        if np.max(np.abs(ci)) < 1e-10:
            return None, (neleca, nelecb)

        # Keep ci as a 2D array (addons already return 2D, but this guards odd inputs)
        ci = np.asarray(ci)

    return ci, (neleca, nelecb)

if __name__ == "__main__":
    # Apply: a_{2,alpha}  c^†_{5,beta}  a_{1,beta}   (in this order on |psi>)
    # parameters
    norb = 2
    neleca, nelecb = 1, 0

    # build CI vector: |10>
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = np.zeros((na, nb))
    ci0[0, 0] = 1.0   # |10> has address 0

    print("Initial CI:")
    print(ci0)

    # apply a_{0,alpha}
    ops = [('ann', 0, 'alpha')]

    ci1, nelec1 = apply_operator_string_fci(ci0, norb, (neleca, nelecb), ops)

    print("After a_{0,alpha}:")
    print(ci1)
    print("New electron numbers:", nelec1)
