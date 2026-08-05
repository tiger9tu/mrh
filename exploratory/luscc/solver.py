from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver
from mrh.my_pyscf.lassi.op_o1.frag import FragTDMInt
from mrh.my_pyscf.lassi.op_o1.utilities import fermion_spin_shuffle
import numpy as np
from pyscf import lib
from pyscf.lib.numpy_helper import tag_array
from pyscf.scf.addons import canonical_orth_ as _pyscf_canonical_orth
from pyscf.fci.direct_spin1 import trans_rdm12s as _fci_tdm12s
from pyscf.csf_fci import csf_solver
from .excitations import apply_operator_string_fci
from copy import deepcopy
import os


# ── Optimized FragTDMInt subclass ──────────────────────────────────────────────

class LSIFragTDMInt(FragTDMInt):
    """FragTDMInt subclass that reuses precomputed spectator TDMs for within-group pairs.

    For two excited states A|LAS_{r,j1}> and A|LAS_{r,j2}> from the same (A, r)
    group and a spectator fragment fi, the transition density matrix between them
    equals the reference intra-rootspace TDM dm1[r][r][k1, k2].  Instead of
    recomputing these with trans_rdm12s, we look them up from a precomputed block
    built once per (fi, r_pool) pair.

    The cache is stored on the LAS object as ``las._lsi_tdm_cache`` and has the
    structure::

        {
          'spectator_skip': {fi: [(rs_a, rs_b), ...]},
          'pair_tdms': {(rs_a, rs_b): {fi: (dm1_1x1, dm2_1x1, ovlp_1x1)}},
        }
    """

    def __init__(self, las, ci, hopping_index, zerop_index, onep_index, norb, nroots, nelec_rs,
                 rootaddr, fragaddr, idx_frag, mask_ints, **kwargs):
        cache = getattr(las, '_lsi_tdm_cache', None)
        if cache is not None:
            skip = cache.get('spectator_skip', {}).get(idx_frag, [])
            if skip:
                mask_ints = mask_ints.copy()
                for rs_a, rs_b in skip:
                    mask_ints[rs_a, rs_b] = False
                    mask_ints[rs_b, rs_a] = False
        super().__init__(las, ci, hopping_index, zerop_index, onep_index, norb, nroots, nelec_rs,
                         rootaddr, fragaddr, idx_frag, mask_ints, **kwargs)
        if cache is not None:
            self._fill_from_cache(cache)

    def _fill_from_cache(self, cache):
        fi = self.idx_frag
        for (rs_a, rs_b), frag_data in cache.get('pair_tdms', {}).items():
            if fi not in frag_data:
                continue
            dm1_val, dm2_val, ovlp_val = frag_data[fi]
            ir = self.unique_root[rs_a]
            jr = self.unique_root[rs_b]
            if ir < jr:
                ir, jr = jr, ir
            if ir == jr:
                continue  # degenerate unique roots; skip to avoid corruption
            if self.dm1[ir][jr] is None:
                self.dm1[ir][jr] = np.ascontiguousarray(dm1_val)
            if dm2_val is not None and self.dm2[ir][jr] is None:
                self.dm2[ir][jr] = np.ascontiguousarray(dm2_val)
            if self.ovlp[ir][jr] is None:
                self.ovlp[ir][jr] = np.ascontiguousarray(ovlp_val)
                self.ovlp[jr][ir] = np.ascontiguousarray(ovlp_val.conj().T)


# ── Main solver class ─────────────────────────────────────────────────────────

class LUSCC(LASSI):
    """Perform LUSCC using the LASSI framework.

    Supports two use cases (unified):

    las-luscc (special case):
        LUSCC(las, a_idxs, i_idxs)
        Single LAS reference state.  Excitations Aᵢ are applied to |las⟩.

    lsi-luscc (general case):
        LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=0.01)
        LASSI reference.  |lsi⟩ = Σⱼ cⱼ|las_j⟩.  Significant components
        (|cⱼ| > threshold) are used as reference states; each excitation Aᵢ is
        applied to every significant component.

    Args:
        las_or_lsi : LAS object *or* post-kernel LASSI object
        a_idxs     : list of creation-operator index tuples (pre-selected)
        i_idxs     : list of annihilation-operator index tuples (pre-selected)
        state      : which LASSI eigenstate to read SI coefficients from (default 0)
        threshold  : |SI coefficient| cutoff for selecting significant LAS
                     components (default 0.01)
        smult_si   : target total spin multiplicity for the SI diagonalization.
                     If set, Davidson diagonalization is used by default because
                     the in-core path does not enforce the target spin sector.
    """

    def __init__(self, las_or_lsi, a_idxs, i_idxs,
                 state=0, threshold=0.01, top_m=None, lindep_thresh=None, norm_thresh=None,
                 opt=1, smult_si=None, internally_contracted=False,
                 internal_backend="raw", share_spectator_ci=False,
                 spin_complete=True, **kwargs):
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self._smult_si = smult_si
        self._internally_contracted = internally_contracted
        self._spin_complete = bool(spin_complete)
        if internal_backend not in (
                "raw", "matrix_free_o1", "matrix_free_o1_iterative"):
            raise ValueError(
                "internal_backend must be 'raw', 'matrix_free_o1', or "
                "'matrix_free_o1_iterative'")
        self._internal_backend = internal_backend
        self._share_spectator_ci = bool(share_spectator_ci)
        self._n_ref_rs = None
        self._ref_product_indices = []
        self._exc_rs_meta = []
        self._lsi_tdm_cache = None
        self._fragint_class = None
        self._norm_thresh = norm_thresh if norm_thresh is not None else 1e-12

        if isinstance(las_or_lsi, LASSI):
            # lsi-luscc: LASSI object provided
            self._ref_lsi = las_or_lsi
            self._lsi_state = state
            self._si_threshold = threshold
            self._top_m = top_m  # if set, select top-m by |ci| regardless of threshold
            las = las_or_lsi._las
            # lsi-luscc generates many near-degenerate states; tighter lindep
            # threshold is needed to avoid a near-singular orthogonal basis.
            self._lindep_thresh = lindep_thresh if lindep_thresh is not None else 1e-3
        else:
            # las-luscc: bare LAS object (single-root special case)
            self._ref_lsi = None
            self._top_m = None
            self._lindep_thresh = lindep_thresh if lindep_thresh is not None else 1e-5
            las = las_or_lsi

        LASSI.__init__(self, las, opt=opt, **kwargs)

    # ------------------------------------------------------------------
    # Significant component selection
    # ------------------------------------------------------------------

    def _select_sig_indices(self):
        """Return indices of significant LAS components for the reference state."""
        if self._ref_lsi is not None and self._ref_lsi.si is not None:
            si_vec = self._ref_lsi.si[:, self._lsi_state]
            if self._top_m is not None:
                # Count-based: select top-m by |ci|
                m = min(self._top_m, len(si_vec))
                sig_indices = np.argsort(-np.abs(si_vec))[:m].tolist()
            else:
                sig_indices = [j for j in range(len(si_vec))
                               if abs(si_vec[j]) > self._si_threshold]
                if len(sig_indices) == 0:
                    # Fall back to dominant component
                    sig_indices = [int(np.argmax(np.abs(si_vec)))]
        else:
            # Single-root case: only root 0
            sig_indices = [0]
        return sig_indices

    # ------------------------------------------------------------------
    # Operator application
    # ------------------------------------------------------------------

    def getAci(self, a_idx, i_idx, ref_ci=None, ref_nelecas_sub=None,
               return_norm_phase=False):
        """Apply excitation operator to a reference CI vector.

        A|ψ⟩ = a₀a₁…i₁i₀|ψ⟩

        Args:
            a_idx            : creation operator indices (spinless)
            i_idx            : annihilation operator indices (spinless)
            ref_ci           : list of per-fragment CI vectors to act on.
                               Defaults to root 0 of self.ci (original behaviour).
            ref_nelecas_sub  : list of (neleca, nelecb) per fragment for ref_ci.
                               Defaults to self.nelecas_sub.

        Returns:
            (Aci, nelecas_sub) or (None, None) if the excitation is invalid.
        """
        frag_orbs_start = [0]
        for norb_f in self.ncas_sub[:-1]:
            frag_orbs_start.append(frag_orbs_start[-1] + norb_f)

        if ref_ci is None:
            ref_ci = [frag_ci[0] for frag_ci in self.ci]
        if ref_nelecas_sub is None:
            ref_nelecas_sub = self.nelecas_sub

        # Excitation operators replace only the CI arrays of fragments they
        # touch. Spectator arrays are immutable, so an optional shallow list
        # copy preserves their identity and lets MRH recognize equivalent
        # rootspaces without duplicating the underlying CI storage.
        Aci = list(ref_ci) if self._share_spectator_ci else deepcopy(ref_ci)
        frag_ops = [[] for _ in range(self.nfrags)]
        ordered_ops = []

        for op_type, idx_list in [('ann', i_idx), ('cre', a_idx[::-1])]:
            for idx in idx_list:
                spatial = idx % self.ncas
                spin = idx // self.ncas
                frag_idx = np.searchsorted(frag_orbs_start, spatial, side='right') - 1
                idx_in_frag = spatial - frag_orbs_start[frag_idx]
                frag_ops[frag_idx].append((op_type, idx_in_frag, spin))
                ordered_ops.append((op_type, frag_idx, spin))

        nelecas_sub = deepcopy(list(ref_nelecas_sub))
        raw_norm = 1.0
        for fi, ci_f in enumerate(Aci):
            if len(frag_ops[fi]) == 0:
                continue
            ci_f_new, (neleca, nelecb) = apply_operator_string_fci(
                ci_f, self.ncas_sub[fi], ref_nelecas_sub[fi], frag_ops[fi])
            if ci_f_new is None or np.all(ci_f_new == 0):
                if return_norm_phase:
                    return None, None, 0.0, 0
                return None, None
            fragment_norm = np.linalg.norm(ci_f_new.ravel())
            raw_norm *= fragment_norm
            ci_f_new = ci_f_new / fragment_norm
            Aci[fi] = ci_f_new
            nelecas_sub[fi] = (neleca, nelecb)

        if return_norm_phase:
            # Convert the globally ordered spin-orbital operator into the
            # fragment-product convention used by the normalized CI factors.
            # The initial/final spin shuffles connect spin-major LASSI basis
            # states to fragment-major states; the middle parity is the
            # Jordan-Wigner string accumulated while applying the operators.
            current = [sum(nelec) for nelec in ref_nelecas_sub]
            cross_phase = 1
            for op_type, frag_idx, _spin in ordered_ops:
                cross_phase *= (-1) ** sum(current[:frag_idx])
                current[frag_idx] += 1 if op_type == 'cre' else -1
            ref_phase = fermion_spin_shuffle(
                [nelec[0] for nelec in ref_nelecas_sub],
                [nelec[1] for nelec in ref_nelecas_sub])
            final_phase = fermion_spin_shuffle(
                [nelec[0] for nelec in nelecas_sub],
                [nelec[1] for nelec in nelecas_sub])
            return Aci, nelecas_sub, raw_norm, ref_phase * cross_phase * final_phase
        return Aci, nelecas_sub

    def _get_active_frags(self, a_idx, i_idx):
        """Return frozenset of fragment indices touched by operator (a_idx, i_idx)."""
        frag_orbs_start = [0]
        for norb_f in self.ncas_sub[:-1]:
            frag_orbs_start.append(frag_orbs_start[-1] + norb_f)
        active = set()
        for idx in list(a_idx) + list(i_idx):
            spatial = idx % self.ncas
            fi = int(np.searchsorted(frag_orbs_start, spatial, side='right') - 1)
            active.add(fi)
        return frozenset(active)

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------

    def prepare_states_(self):
        from mrh.my_pyscf.mcscf.lasci import get_space_info
        from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr

        sig_indices = self._select_sig_indices()
        n_sig = len(sig_indices)
        max_nroots = n_sig * (1 + 2 * len(self.a_idxs))

        charges = np.zeros((max_nroots, self.nfrags), dtype=np.int32)
        spins   = np.zeros((max_nroots, self.nfrags), dtype=np.int32)
        smults  = np.ones ((max_nroots, self.nfrags), dtype=np.int32)
        wfnsyms = np.zeros((max_nroots, self.nfrags), dtype=np.int32)

        if self._ref_lsi is not None:
            # lsi-luscc: sig_indices are product state indices (0..nprod-1).
            # Use get_rootaddr_fragaddr to map each product state j to its
            # rootspace r and per-fragment CI index ki.
            ref = self._ref_lsi
            _charges, _spins, _smults, _wfnsyms = get_space_info(ref)
            nelec_frs = self.get_nelec_frs(ref)
            lroots = get_lroots(ref.ci)
            rootaddr, fragaddr = get_rootaddr_fragaddr(lroots)

            def _ref_ci_and_nelec(j):
                r = rootaddr[j]
                ref_ci = []
                for fi in range(self.nfrags):
                    ki = fragaddr[fi, j]
                    ci_fir = ref.ci[fi][r]
                    ref_ci.append(ci_fir[ki] if ci_fir.ndim > 2 else ci_fir)
                ref_nelec = [tuple(nelec_frs[fi, r]) for fi in range(self.nfrags)]
                return ref_ci, ref_nelec, r
        else:
            # las-luscc: sig_indices are simple LAS root indices.
            _charges, _spins, _smults, _wfnsyms = get_space_info(self._las)
            nelec_frs = self.get_nelec_frs(self._las)
            original_ci = self.ci
            fragaddr = None  # not used for las-luscc (no within-group pairs)

            def _ref_ci_and_nelec(j):
                ref_ci = [original_ci[fi][j] for fi in range(self.nfrags)]
                ref_nelec = [tuple(nelec_frs[fi, j]) for fi in range(self.nfrags)]
                return ref_ci, ref_nelec, j

        # ── Group reference states by rootspace r ─────────────────────────
        # Multiple sig_indices j can map to the same rootspace r when the
        # source LASSI object has lroots > 1 for that rootspace.  Using the
        # full ref.ci[fi][r] array (already LASCI-orthogonalised per frag)
        # as a single multi-root entry lets op_o1 vectorise over lroots pairs
        # without creating spurious Cartesian-product states, because LASCI
        # guarantees that per-fragment CI vectors within a rootspace are
        # independent bases — the Cartesian product IS the correct product
        # state expansion for that rootspace.
        ref_rs_seen = {}   # r -> pool_idx of first occurrence
        ref_product_indices = []
        new_ci = [[] for _ in range(self.nfrags)]
        self.nroots = 0

        for j in sig_indices:
            _, _, r = _ref_ci_and_nelec(j)
            if r in ref_rs_seen:
                continue  # already added this rootspace's full CI
            ref_rs_seen[r] = self.nroots
            if self._ref_lsi is not None:
                ref_product_indices.extend(np.where(rootaddr == r)[0].tolist())
            else:
                ref_product_indices.append(int(r))
            for fi in range(self.nfrags):
                new_ci[fi].append(ref.ci[fi][r] if self._ref_lsi is not None
                                  else original_ci[fi][r])
            charges[self.nroots] = _charges[r]
            spins  [self.nroots] = _spins  [r]
            smults [self.nroots] = _smults [r]
            wfnsyms[self.nroots] = _wfnsyms[r]
            self.nroots += 1

        n_ref_rs = self.nroots
        self._n_ref_rs = n_ref_rs
        self._ref_product_indices = ref_product_indices
        self._exc_rs_meta = []

        # ── Excited states: one rootspace per valid (A, j) combination ────
        # A|LAS_j⟩ states from different reference product states are
        # *correlated*: the per-fragment CI vectors from different j's cannot
        # be stacked into a multi-root array without creating spurious
        # Cartesian-product states (because the fragment CIs come from
        # different j's and are not jointly LASCI-orthogonalised).
        # We therefore keep one rootspace per excited state, same as before.
        for operator_index, (a_idx, i_idx) in enumerate(zip(self.a_idxs, self.i_idxs)):
            active_frags = self._get_active_frags(a_idx, i_idx)
            for j in sig_indices:
                ref_ci_j, ref_nelecas_sub_j, r_source = _ref_ci_and_nelec(j)
                r_pool = ref_rs_seen.get(r_source)

                for dir_idx, (a, i) in enumerate([(a_idx, i_idx), (i_idx, a_idx)]):
                    Aci, nelecas_sub_new, raw_norm, operator_phase = self.getAci(
                        a, i, ref_ci=ref_ci_j, ref_nelecas_sub=ref_nelecas_sub_j,
                        return_norm_phase=True)
                    if Aci is not None:
                        # Track metadata for the within-group TDM optimization
                        if fragaddr is not None and r_pool is not None:
                            ki_per_frag = [int(fragaddr[fi, j]) for fi in range(self.nfrags)]
                        else:
                            ki_per_frag = [0] * self.nfrags
                        self._exc_rs_meta.append({
                            'group_key': (tuple(a_idx), tuple(i_idx), r_pool, dir_idx),
                            'active_frags': active_frags,
                            'ref_rs_pool': r_pool,
                            'ki_per_frag': ki_per_frag,
                            'nelecas': [tuple(nelecas_sub_new[fi]) for fi in range(self.nfrags)],
                            'operator_index': operator_index,
                            'source_product_index': int(j),
                            'direction': dir_idx,
                            'raw_norm': float(raw_norm),
                            'operator_phase': int(operator_phase),
                        })

                        for fi, ci_f in enumerate(Aci):
                            new_ci[fi].append(ci_f)
                            charges[self.nroots, fi] = (
                                self.ncas_sub[fi] - sum(nelecas_sub_new[fi]))
                            spins  [self.nroots, fi] = (
                                nelecas_sub_new[fi][0] - nelecas_sub_new[fi][1])
                            smults [self.nroots, fi] = (
                                abs(spins[self.nroots, fi]) + 1)
                        self.nroots += 1

        self.e_states_meaningless = True

        from mrh.my_pyscf.lassi.citools import get_lroots as _get_lroots
        lroots_new = _get_lroots(new_ci)
        n_total = int(np.sum(np.prod(lroots_new, axis=0)))
        n_exc_states = n_total - int(np.sum(np.prod(lroots_new[:, :n_ref_rs], axis=0)))
        lib.logger.info(self,
            'nroots prepared: %d states (%d reference + %d excited) '
            'in %d rootspaces (%d ref + %d exc; avg %.1f states/rootspace)',
            n_total, n_total - n_exc_states, n_exc_states,
            self.nroots, n_ref_rs, self.nroots - n_ref_rs,
            n_total / max(self.nroots, 1))

        charges = charges[:self.nroots]
        spins   = spins  [:self.nroots]
        smults  = smults [:self.nroots]
        wfnsyms = wfnsyms[:self.nroots]

        self.ci = new_ci
        self.weights = np.zeros(self.nroots)
        self.weights[:n_ref_rs] = 1.0

        self.fciboxes = [get_h1e_zipped_fcisolver(state_average_n_mix(
            self._las, [csf_solver(self._las.mol, smult=s2p1).set(
                charge=c, spin=m2, wfnsym=ir)
                for c, m2, s2p1, ir in zip(c_r, m2_r, s2p1_r, ir_r)],
            self.weights).fcisolver)
            for c_r, m2_r, s2p1_r, ir_r in zip(
                charges.T, spins.T, smults.T, wfnsyms.T)]

    def _build_internal_contraction(self, sparse=False):
        """Map the raw LAS-product basis onto {|lsi'>, (T-T†)|lsi'>}.

        ``prepare_states_`` normalizes each excited fragment product
        independently.  Recover the physical operator action with its recorded
        raw norm and fermionic phase, and retain one column per selected
        anti-Hermitian generator.
        """
        from mrh.my_pyscf.lassi.citools import get_lroots

        if self._ref_lsi is None:
            ref_coeff = np.ones(1)
        else:
            ref_coeff = np.asarray(
                self._ref_lsi.si[:, self._lsi_state]).reshape(-1)

        lroots = get_lroots(self.ci)
        nprods_r = np.prod(lroots, axis=0).astype(int)
        offsets = np.append(0, np.cumsum(nprods_r))
        nraw = int(offsets[-1])
        nref_raw = int(offsets[self._n_ref_rs])
        if nref_raw != len(self._ref_product_indices):
            raise RuntimeError(
                "internally contracted reference-address mismatch: "
                f"{nref_raw} raw states != {len(self._ref_product_indices)} products")

        shape = (nraw, 1 + len(self.a_idxs))
        dtype = np.result_type(ref_coeff.dtype, np.float64)
        if sparse:
            from scipy.sparse import coo_matrix

            rows = []
            columns = []
            values = []

            def add(row, column, value):
                if value != 0:
                    rows.append(row)
                    columns.append(column)
                    values.append(value)
        else:
            transform = np.zeros(shape, dtype=dtype)

            def add(row, column, value):
                transform[row, column] += value

        for raw_idx, product_idx in enumerate(self._ref_product_indices):
            add(raw_idx, 0, ref_coeff[product_idx])

        if len(self._exc_rs_meta) != self.nroots - self._n_ref_rs:
            raise RuntimeError("excited-rootspace metadata is incomplete")
        for exc_idx, meta in enumerate(self._exc_rs_meta):
            rootspace = self._n_ref_rs + exc_idx
            if nprods_r[rootspace] != 1:
                raise RuntimeError(
                    "internally contracted excited rootspaces must contain one product state")
            row = int(offsets[rootspace])
            product_idx = meta["source_product_index"]
            direction_sign = 1 if meta["direction"] == 0 else -1
            add(row, 1 + meta["operator_index"], (
                ref_coeff[product_idx]
                * meta["raw_norm"]
                * meta["operator_phase"]
                * direction_sign
            ))

        if sparse:
            transform = coo_matrix(
                (np.asarray(values, dtype=dtype), (rows, columns)),
                shape=shape,
            ).tocsr()
            transform.sum_duplicates()

        # A selected generator may annihilate every retained reference
        # component.  Such a tangent is exactly zero and is not part of the
        # contracted model.
        if sparse:
            norms = np.sqrt(
                np.asarray(abs(transform).power(2).sum(axis=0)).ravel())
        else:
            norms = np.linalg.norm(transform, axis=0)
        keep = norms > self._norm_thresh
        keep[0] = True
        dropped = int(np.count_nonzero(~keep))
        transform = transform[:, keep]
        lib.logger.info(
            self,
            "Internally contracted LUSCC basis: %d raw states -> %d states "
            "(1 reference + %d tangents; %d null generators dropped)",
            nraw, transform.shape[1], transform.shape[1] - 1, dropped)
        return transform

    def _kernel_internally_contracted(self, **kwargs):
        """Contract H/S/S2 and solve the small generalized eigenvalue problem."""
        from scipy import linalg
        from mrh.my_pyscf.lassi import op_o0, op_o1
        from mrh.my_pyscf.lassi.lassi import las_symm_tuple

        if self._smult_si is not None or kwargs.get("smult_si") is not None:
            raise ValueError(
                "internally contracted LUSCC currently discovers spin; do not set smult_si")

        iterative_internal = (
            self._internal_backend == "matrix_free_o1_iterative")
        transform = self._build_internal_contraction(
            sparse=iterative_internal)
        e0, h1, h2 = self.ham_2q(
            mo_coeff=kwargs.get("mo_coeff"),
            veff_c=kwargs.get("veff_c"),
            h2eff_sub=kwargs.get("h2eff_sub"),
            soc=0)
        nelec_frs = self.get_nelec_frs()
        opt = kwargs.get("opt", self.opt)
        if opt not in (0, 1):
            raise ValueError(f"unsupported LASSI contraction backend opt={opt}")
        chkkey = self.get_o1_chk_key() if callable(
            getattr(self, "get_o1_chk_key", None)) else None
        if self._internal_backend in (
                "matrix_free_o1", "matrix_free_o1_iterative"):
            if opt != 1:
                raise ValueError(
                    f"{self._internal_backend} is an opt=1 factorized "
                    "operator backend; "
                    "set opt=1")
            h_op, s2_op, ovlp_op, _, get_raw_ovlp = (
                op_o1.gen_contract_op_si_hdiag(
                self, h1, h2, self.ci, nelec_frs, smult_fr=None, soc=0,
                chkfile=getattr(self, "chkfile", None), chkkey=chkkey))

            # Stream contracted vectors without ever forming an nraw x nraw
            # Hamiltonian.  The dense variant builds requested k x k
            # matrices; the iterative variant builds only the overlap metric
            # and solves the ground root through Hamiltonian matvecs.
            ncontract = transform.shape[1]
            dtype = np.result_type(transform.dtype, h1.dtype, h2.dtype)
            transform_h = transform.conj().T
            if iterative_internal:
                # The factorized overlap LinearOperator is optimized for
                # Davidson-sized trial blocks, not thousands of basis
                # columns.  Its parent exposes an exact direct overlap
                # constructor that is cheap relative to H and bounded by the
                # contracted-space O(k^2) metric we need in any case.
                raw_ovlp = get_raw_ovlp()
                transform_dense = transform.toarray()
                sc = transform_dense.conj().T @ raw_ovlp @ transform_dense
                sc = (sc + sc.conj().T) / 2
                lib.logger.info(
                    self,
                    "Direct overlap metric: %d raw states -> %d contracted "
                    "states",
                    raw_ovlp.shape[0], ncontract)
                raw2orth = _pyscf_canonical_orth(
                    sc, thr=self._lindep_thresh)
                north = raw2orth.shape[1]
                if north == 0:
                    raise RuntimeError(
                        "internally contracted LUSCC basis is linearly "
                        "dependent")

                def horth_matvec(vector):
                    contracted = raw2orth @ vector
                    raw = transform @ contracted
                    hraw = h_op.matvec(raw)
                    return raw2orth.conj().T @ (transform_h @ hraw)

                if north == 1:
                    coeff_orth = np.ones((1, 1), dtype=dtype)
                    e = np.asarray(
                        [np.real(horth_matvec(coeff_orth[:, 0])[0])])
                else:
                    from scipy.sparse.linalg import LinearOperator, eigsh

                    horth_op = LinearOperator(
                        (north, north), matvec=horth_matvec, dtype=dtype)
                    reference = np.zeros(ncontract, dtype=dtype)
                    reference[0] = 1
                    v0 = raw2orth.conj().T @ (sc @ reference)
                    v0_norm = np.linalg.norm(v0)
                    if v0_norm:
                        v0 /= v0_norm
                    else:
                        v0 = np.ones(north, dtype=dtype)
                        v0 /= np.linalg.norm(v0)
                    iterative_tol = kwargs.get(
                        "iterative_tol",
                        min(1e-10, getattr(self, "conv_tol", 1e-8)))
                    iterative_maxiter = kwargs.get(
                        "iterative_maxiter", max(1000, 5 * north))
                    e, coeff_orth = eigsh(
                        horth_op, k=1, which="SA", v0=v0,
                        tol=iterative_tol, maxiter=iterative_maxiter)
                    order = np.argsort(e)
                    e = np.real(e[order])
                    coeff_orth = coeff_orth[:, order]
                coeff_contract = raw2orth @ coeff_orth
                si = transform @ coeff_contract
                s2 = np.asarray([
                    np.real(np.vdot(si[:, root], s2_op.matvec(si[:, root])))
                    for root in range(si.shape[1])
                ])
                e = np.real(e) + e0
            else:
                hc = np.empty((ncontract, ncontract), dtype=dtype)
                sc = np.empty_like(hc)
                s2c = np.empty_like(hc)
                for column in range(ncontract):
                    ket = transform[:, column]
                    hc[:, column] = transform_h @ h_op.matvec(ket)
                    sc[:, column] = transform_h @ ovlp_op.matvec(ket)
                    s2c[:, column] = transform_h @ s2_op.matvec(ket)
                    if column == 0 or (column + 1) % 10 == 0:
                        lib.logger.info(
                            self,
                            "Matrix-free internal contraction: %d/%d columns",
                            column + 1, ncontract)
        else:
            op = (op_o0, op_o1)[opt]
            ham, s2mat, ovlp, _ = op.ham(
                self, h1, h2, self.ci, nelec_frs, smult_fr=None, soc=0,
                chkfile=getattr(self, "chkfile", None), chkkey=chkkey)
            hc = transform.conj().T @ ham @ transform
            sc = transform.conj().T @ ovlp @ transform
            s2c = transform.conj().T @ s2mat @ transform
        if not iterative_internal:
            hc = (hc + hc.conj().T) / 2
            sc = (sc + sc.conj().T) / 2
            s2c = (s2c + s2c.conj().T) / 2

            raw2orth = _pyscf_canonical_orth(sc, thr=self._lindep_thresh)
            if raw2orth.shape[1] == 0:
                raise RuntimeError(
                    "internally contracted LUSCC basis is linearly dependent")
            horth = raw2orth.conj().T @ hc @ raw2orth
            e, coeff_orth = linalg.eigh(horth)
            coeff_contract = raw2orth @ coeff_orth
            si = transform @ coeff_contract
            s2 = np.real(np.einsum(
                "ki,kl,li->i", coeff_contract.conj(), s2c, coeff_contract))
            e = np.real(e) + e0

        statesym, _ = las_symm_tuple(self)
        unique_sym = sorted(set(statesym))
        if len(unique_sym) != 1:
            raise RuntimeError(
                "internally contracted LUSCC currently requires one global symmetry block")
        rootsym = np.asarray([unique_sym[0]] * len(e))
        nelec = [tuple(sym[:2]) for sym in rootsym]
        wfnsym = [sym[-1] for sym in rootsym]
        self.e_roots = e
        self.s2 = s2
        self.nelec = nelec
        self.wfnsym = wfnsym
        self.rootsym = rootsym
        self.si = tag_array(
            si, s2=s2, nelec=nelec, wfnsym=wfnsym, rootsym=rootsym,
            break_symmetry=False, soc=False)
        if getattr(self, "sisolver", None) is not None:
            self.sisolver.converged = True
        return self.e_roots, self.si

    # ------------------------------------------------------------------
    # Within-group spectator TDM precomputation (Optimization 2)
    # ------------------------------------------------------------------

    def _compute_spectator_block(self, fi, ref_rs_pool, nelec_fi):
        """Compute the full M_r × M_r TDM block for spectator fragment fi.

        Args:
            fi          : fragment index
            ref_rs_pool : LSI-LUSCC rootspace index for the source reference rootspace
            nelec_fi    : (neleca, nelecb) electron count for fragment fi

        Returns:
            (dm1_block, dm2_block, ovlp_block) with shapes
            (M_r, M_r, 2, norb, norb), (M_r, M_r, 4, norb, norb, norb, norb), (M_r, M_r)
        """
        ci_block = self.ci[fi][ref_rs_pool]          # 3D (M_r, na, nb) or 2D (na, nb)
        if ci_block.ndim == 2:
            ci_block = ci_block[None, :]             # → (1, na, nb)
        M_r = ci_block.shape[0]
        norb_fi = self.ncas_sub[fi]
        na, nb = ci_block.shape[1], ci_block.shape[2]

        dm1_block  = np.zeros((M_r, M_r, 2, norb_fi, norb_fi))
        dm2_block  = np.zeros((M_r, M_r, 4, norb_fi, norb_fi, norb_fi, norb_fi))
        ovlp_block = np.zeros((M_r, M_r))

        for k_a in range(M_r):
            bra = ci_block[k_a].reshape(na, nb)
            for k_b in range(M_r):
                ket = ci_block[k_b].reshape(na, nb)
                ovlp_block[k_a, k_b] = np.dot(bra.ravel().conj(), ket.ravel())
                d1s, d2s = _fci_tdm12s(bra, ket, norb_fi, nelec_fi)
                dm1_block[k_a, k_b] = np.stack(d1s, axis=0).transpose(0, 2, 1)
                dm2_block[k_a, k_b] = np.stack(d2s, axis=0)

        return dm1_block, dm2_block, ovlp_block

    def _build_lsi_tdm_cache(self):
        """Precompute spectator TDMs for within-group pairs (Optimization 2).

        Within each (operator, reference-rootspace, direction) group of excited
        states, spectator-fragment TDMs are identical to reference intra-rootspace
        TDMs already available from the reference CI arrays.  This method computes
        those blocks once per (fragment, reference-rootspace) pair and caches the
        per-pair slices so LSIFragTDMInt can skip the corresponding trans_rdm12s
        calls during _init_crunch_.

        Returns None if there are no within-group pairs to optimise (e.g. all
        groups have only one member).
        """
        if not self._exc_rs_meta:
            return None

        n_ref_rs = self._n_ref_rs

        # Group excited rootspaces by group_key
        groups = {}
        for exc_idx, meta in enumerate(self._exc_rs_meta):
            gk = meta['group_key']
            rs_exc = n_ref_rs + exc_idx
            if gk not in groups:
                groups[gk] = {
                    'rs_list': [],
                    'ki_per_frag_list': [],
                    'nelecas_list': [],
                    'active_frags': meta['active_frags'],
                    'ref_rs_pool': meta['ref_rs_pool'],
                }
            groups[gk]['rs_list'].append(rs_exc)
            groups[gk]['ki_per_frag_list'].append(meta['ki_per_frag'])
            groups[gk]['nelecas_list'].append(meta['nelecas'])

        spectator_skip = {}   # fi → [(rs_a, rs_b), ...]
        pair_tdms = {}        # (rs_a, rs_b) → {fi: (dm1_1x1, dm2_1x1, ovlp_1x1)}
        spec_block_cache = {} # (fi, r_pool) → (dm1_block, dm2_block, ovlp_block)
        n_pairs_saved = 0

        for g in groups.values():
            rs_list = g['rs_list']
            if len(rs_list) < 2:
                continue

            active_frags   = g['active_frags']
            ref_rs_pool    = g['ref_rs_pool']
            ki_list        = g['ki_per_frag_list']   # ki_list[state_idx][fi]
            nelecas_list   = g['nelecas_list']        # nelecas_list[state_idx][fi]

            if ref_rs_pool is None:
                continue  # safety: no source rootspace recorded

            for idx_a in range(len(rs_list)):
                for idx_b in range(idx_a):
                    rs_a = rs_list[idx_a]   # rs_a > rs_b (added in order)
                    rs_b = rs_list[idx_b]

                    frag_data = {}
                    for fi in range(self.nfrags):
                        if fi in active_frags:
                            continue  # let parent compute active-fragment TDMs

                        # Ensure the spectator block for (fi, ref_rs_pool) is built
                        cache_key = (fi, ref_rs_pool)
                        if cache_key not in spec_block_cache:
                            nelec_fi = nelecas_list[idx_a][fi]  # same for all states (spectator)
                            spec_block_cache[cache_key] = self._compute_spectator_block(
                                fi, ref_rs_pool, nelec_fi)

                        dm1_blk, dm2_blk, ovlp_blk = spec_block_cache[cache_key]
                        k_a = ki_list[idx_a][fi]
                        k_b = ki_list[idx_b][fi]

                        # Slice out the (1,1,...) sub-block for this specific pair
                        dm1_val  = dm1_blk [k_a:k_a+1, k_b:k_b+1]
                        dm2_val  = dm2_blk [k_a:k_a+1, k_b:k_b+1]
                        ovlp_val = ovlp_blk[k_a:k_a+1, k_b:k_b+1]

                        frag_data[fi] = (dm1_val, dm2_val, ovlp_val)
                        spectator_skip.setdefault(fi, []).append((rs_a, rs_b))

                    if frag_data:
                        pair_tdms[(rs_a, rs_b)] = frag_data
                        n_pairs_saved += len(frag_data)

        if not pair_tdms:
            return None

        n_spec_blocks = len(spec_block_cache)
        lib.logger.info(self,
            'LSI-LUSCC TDM cache: %d within-group pairs optimised across %d spectator '
            'fragment-rootspace blocks (%.0f trans_rdm12s calls avoided, '
            '%d blocks computed once)',
            len(pair_tdms), n_spec_blocks, n_pairs_saved, n_spec_blocks)

        return {'spectator_skip': spectator_skip, 'pair_tdms': pair_tdms}

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _spin_complete_states_(self):
        """Close the prepared LAS-LUSCC model under local-Sz spin shuffles.

        Gradient selection acts on spin-orbital generators and can retain only
        part of a local-spin manifold.  A spin-targeted SISolver requires the
        corresponding local-Sz partners so that its orthogonal basis carries a
        well-defined total spin.  Generate those partners with MRH's native
        spin-rotation machinery without changing the selected generator seeds.

        Prepared LUSCC spaces can contain several distinct CI vectors with the
        same charge/spin/smult labels.  ``spaces.spin_shuffle`` treats those
        labels as unique state identifiers and therefore rejects or collapses
        such spaces.  Rotate each prepared CI rootspace independently instead;
        duplicate labels remain distinct basis vectors, as required.
        """
        from mrh.my_pyscf.lassi.spaces import SingleLASRootspace
        from mrh.my_pyscf.lassi.citools import get_lroots
        from mrh.my_pyscf.mcscf.lasci import get_space_info

        charges, spins, smults, wfnsyms = get_space_info(self)
        nroots_before = int(self.nroots)
        nstates_before = int(np.sum(np.prod(get_lroots(self.ci), axis=0)))
        out_charges, out_spins, out_smults, out_wfnsyms = [], [], [], []
        out_weights = []
        out_ci = [[] for _ in range(self.nfrags)]
        for iroot in range(nroots_before):
            ci_ref = [self.ci[ifrag][iroot] for ifrag in range(self.nfrags)]
            ref = SingleLASRootspace(
                self, spins[iroot], smults[iroot], charges[iroot],
                self.weights[iroot], ci=ci_ref,
                fragsym=wfnsyms[iroot])
            ci_sz = ref.get_ci_szrot()
            for partner in ref.gen_spin_shuffles():
                partner_ci = [
                    ci_sz[ifrag][partner.spins[ifrag]]
                    for ifrag in range(self.nfrags)
                ]
                out_charges.append(partner.charges.copy())
                out_spins.append(partner.spins.copy())
                out_smults.append(partner.smults.copy())
                out_wfnsyms.append(np.asarray(wfnsyms[iroot]).copy())
                out_weights.append(
                    self.weights[iroot]
                    if np.array_equal(partner.spins, spins[iroot]) else 0.0
                )
                for ifrag, ci in enumerate(partner_ci):
                    out_ci[ifrag].append(ci)

        # Build the solver metadata with no cached CI so state_average does not
        # try to match duplicate quantum-label rows.  The independently rotated
        # vectors are installed immediately afterwards.
        las_seed = self._las.state_average(
            weights=np.ones(1), charges=charges[:1], spins=spins[:1],
            smults=smults[:1], wfnsyms=wfnsyms[:1],
            assert_no_dupes=False)
        las_seed.ci = None
        las_complete = las_seed.state_average(
            weights=np.asarray(out_weights), charges=np.asarray(out_charges),
            spins=np.asarray(out_spins), smults=np.asarray(out_smults),
            wfnsyms=np.asarray(out_wfnsyms), assert_no_dupes=False)
        las_complete.ci = out_ci
        las_complete.converged = self.converged

        self.ci = las_complete.ci
        self.fciboxes = las_complete.fciboxes
        self.weights = np.asarray(las_complete.weights)
        self.nroots = int(las_complete.nroots)
        nstates_after = int(np.sum(np.prod(get_lroots(self.ci), axis=0)))
        self.spin_completion_counts = {
            "rootspaces_before": nroots_before,
            "rootspaces_after": self.nroots,
            "model_states_before": nstates_before,
            "model_states_after": nstates_after,
        }
        # Spin shuffling changes rootspace addresses, invalidating the optional
        # spectator-cache metadata.  The unoptimised path remains exact.
        self._exc_rs_meta = []
        lib.logger.info(
            self,
            "Spin-completed LAS-LUSCC model: %d -> %d rootspaces; "
            "%d -> %d model states",
            nroots_before, self.nroots, nstates_before, nstates_after)

    def _filter_smult_roots_(self, smult_si, tol=1e-4):
        target_s = (smult_si - 1) / 2
        target_s2 = target_s * (target_s + 1)
        s2 = np.asarray(self.si.s2)
        idx = np.where(np.abs(s2 - target_s2) <= tol)[0]
        if len(idx) == 0:
            raise RuntimeError(
                f"LSI_LUSCC found no roots with spin multiplicity {smult_si} "
                f"(target <S^2>={target_s2})")
        sisolver = getattr(self, "sisolver", None)
        nroots = getattr(sisolver, "nroots", None)
        if nroots is not None:
            idx = idx[:nroots]

        self.e_roots = self.e_roots[idx]
        si = self.si[:, idx]
        self.s2 = np.asarray(self.si.s2)[idx]
        self.nelec = [self.si.nelec[i] for i in idx]
        self.wfnsym = [self.si.wfnsym[i] for i in idx]
        self.rootsym = np.asarray(self.si.rootsym)[idx]
        self.si = tag_array(
            si, s2=self.s2, nelec=self.nelec, wfnsym=self.wfnsym,
            rootsym=self.rootsym, break_symmetry=self.si.break_symmetry,
            soc=self.si.soc)
        return self.e_roots, self.si

    def kernel(self, **kwargs):
        requested_smult = self._smult_si if self._smult_si is not None else kwargs.get('smult_si')
        injected_davidson = False
        # Newer MRH branches accept ``smult_si`` and provide a spin-coupled
        # Davidson solver.  tiger9tu/mrh master does not; on that API perform
        # the exact direct diagonalization and filter the resulting spin-pure
        # roots by <S^2> below.
        # Spin targeting is an object-level SISolver capability.  Some MRH
        # versions expose ``smult_si`` only on ``LASSI.kernel`` while their
        # module-level ``lassi`` function retains an older signature, so the
        # latter is not a reliable capability probe.
        sisolver = getattr(self, "sisolver", None)
        supports_smult_si = sisolver is not None and hasattr(sisolver, "smult")
        if supports_smult_si:
            if self._smult_si is not None:
                kwargs.setdefault('smult_si', self._smult_si)
            if requested_smult is not None and 'davidson_only' not in kwargs:
                kwargs['davidson_only'] = True
                injected_davidson = True
        else:
            kwargs.pop('smult_si', None)

        self.prepare_states_()
        if self._internally_contracted:
            # The spectator-TDM cache accelerates only the opt=1 operator
            # backend. Building it for opt=0 is pure overhead and can dominate
            # the small internally contracted solve.
            if self.opt == 1:
                cache = self._build_lsi_tdm_cache()
                if cache is not None:
                    self._lsi_tdm_cache = cache
                    self._fragint_class = LSIFragTDMInt
            try:
                return self._kernel_internally_contracted(**kwargs)
            finally:
                self._lsi_tdm_cache = None
                self._fragint_class = None
        # A target-spin solve needs a model closed under local-Sz rotations
        # regardless of whether this MRH version can target spin inside its
        # Davidson solver.  Older versions use the exact direct solver and
        # filter by <S^2> afterwards; without this completion their truncated
        # basis may contain no spin-pure root to retain.
        if requested_smult is not None:
            if self._spin_complete:
                from time import perf_counter
                spin_completion_started = perf_counter()
                self._spin_complete_states_()
                self.spin_completion_elapsed_s = (
                    perf_counter() - spin_completion_started)
            else:
                from mrh.my_pyscf.lassi.citools import get_lroots
                nstates = int(np.sum(np.prod(get_lroots(self.ci), axis=0)))
                self.spin_completion_counts = {
                    "rootspaces_before": int(self.nroots),
                    "rootspaces_after": int(self.nroots),
                    "model_states_before": nstates,
                    "model_states_after": nstates,
                }
                self.spin_completion_elapsed_s = 0.0
                lib.logger.info(
                    self,
                    "Spin completion disabled: retaining %d rootspaces and "
                    "%d model states",
                    self.nroots, nstates)

        # Build spectator TDM cache and activate the optimised FragTDMInt subclass
        cache = self._build_lsi_tdm_cache()
        if cache is not None:
            self._lsi_tdm_cache = cache
            self._fragint_class = LSIFragTDMInt

        import mrh.my_pyscf.lassi.citools as _citools
        import mrh.my_pyscf.lassi.spaces as _spaces

        try:
            import mrh.my_pyscf.lassi.basis as _basis
        except ImportError:
            _basis = None
        try:
            import mrh.my_pyscf.lassi.sisolver as _sisolver
        except ImportError:
            _sisolver = None

        def _safe_canonical_orth(ovlp, thr=1e-7):
            ovlp = np.asarray(ovlp)
            diag = np.real(np.diag(ovlp))
            keep = np.isfinite(diag) & (diag > self._norm_thresh)
            if np.all(keep):
                return _pyscf_canonical_orth(ovlp, thr=thr)

            if np.count_nonzero(keep):
                x_keep = _pyscf_canonical_orth(ovlp[np.ix_(keep, keep)], thr=thr)
                xmat = np.zeros((ovlp.shape[0], x_keep.shape[1]), dtype=x_keep.dtype)
                xmat[keep] = x_keep
            else:
                xmat = np.zeros((ovlp.shape[0], 0), dtype=ovlp.dtype)
            lib.logger.warn(
                self,
                'Dropped %d low-norm LASSI model states before canonical orthogonalization '
                '(norm_thresh=%g)',
                ovlp.shape[0] - np.count_nonzero(keep), self._norm_thresh)
            return xmat

        _thresh_modules = [m for m in (_citools, _basis, _sisolver, _spaces)
                           if m is not None]
        _old_thresh = {}
        for _mod in _thresh_modules:
            if hasattr(_mod, "LINDEP_THRESH"):
                _old_thresh[_mod] = _mod.LINDEP_THRESH
                _mod.LINDEP_THRESH = self._lindep_thresh
        _orth_modules = [m for m in (_basis, _sisolver)
                         if m is not None and hasattr(m, 'canonical_orth_')]
        _old_canonical_orth = {m: m.canonical_orth_ for m in _orth_modules}
        for _mod in _orth_modules:
            _mod.canonical_orth_ = _safe_canonical_orth
        try:
            try:
                result = LASSI.kernel(self, **kwargs)
            except AssertionError:
                if requested_smult is None or not injected_davidson:
                    raise
                if os.environ.get("LUSCC_STRICT_SPIN") == "1":
                    raise
                lib.logger.warn(
                    self,
                    "Spin-coupled Davidson diagonalization is not available for "
                    "this LAS-LUSCC state space; falling back to direct "
                    "diagonalization followed by <S^2> root filtering.")
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop('smult_si', None)
                fallback_kwargs['davidson_only'] = False
                result = LASSI.kernel(self, **fallback_kwargs)
                result = self._filter_smult_roots_(requested_smult)
            if requested_smult is not None and not supports_smult_si:
                result = self._filter_smult_roots_(requested_smult)
        finally:
            for _mod, _thr in _old_thresh.items():
                _mod.LINDEP_THRESH = _thr
            for _mod, _orth in _old_canonical_orth.items():
                _mod.canonical_orth_ = _orth
            self._lsi_tdm_cache = None
            self._fragint_class = None
        return result

    def filter_spaces(self, las):
        return las


# Backward-compatible name used by the original standalone prototype.
LSI_LUSCC = LUSCC
