"""LAS-LUSCC for stilbene along the central-dihedral coordinate.

The active space is split into the two phenyl pi systems and the central
ethylene pi bond.  A small, gradient-selected cluster expansion keeps this
larger demonstration tractable.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import gto, scf

from mrh.exploratory.luscc import LUSCC, get_grad_exact
from mrh.my_pyscf import lassi
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF


def select_largest(a_idxs, i_idxs, gradients, fraction):
    """Return the fraction of excitations with the largest |gradient|."""
    nselect = max(1, int(np.ceil(fraction * len(gradients))))
    order = np.argsort(-np.abs(gradients))[:nselect]
    return [a_idxs[i] for i in order], [i_idxs[i] for i in order]


parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, choices=(1, 60, 90, 120, 180),
                    default=90)
parser.add_argument("--state", choices=("singlet", "triplet"),
                    default="singlet")
parser.add_argument("--fraction", type=float, default=0.01)
parser.add_argument("--geometry-dir", type=Path)
args = parser.parse_args()

if args.geometry_dir is None:
    args.geometry_dir = (Path(__file__).resolve().parents[3]
                         / "lsi-uscc" / "tasks" / "geom")
geometry_tag = "stil001" if args.geometry == 1 else f"stil{args.geometry}"
geometry_path = args.geometry_dir / f"{geometry_tag}.xyz"
xyz = geometry_path.read_text()
is_triplet = args.state == "triplet"

mol = gto.M(
    atom=xyz,
    basis="6-31g",
    spin=2 if is_triplet else 0,
    max_memory=400_000,
    verbose=4,
)
mf = (scf.ROHF(mol) if is_triplet else scf.RHF(mol)).run()

ncas_sub = (4, 2, 4)
nelecas_sub = (4, 2, 4)
frag_atoms = (
    (1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19),
    (0, 7, 14, 20),
    (8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25),
)
spin_sub = (1, 3, 1) if is_triplet else (1, 1, 1)
las = LASSCF(mf, ncas_sub, nelecas_sub, spin_sub=spin_sub)
mo_guess = las.localize_init_guess(frag_atoms, mf.mo_coeff)
las.kernel(mo_guess)

e_lassis, _ = lassi.LASSIS(las).kernel()
gradients, _, a_idxs, i_idxs = get_grad_exact(las)
a_selected, i_selected = select_largest(
    a_idxs, i_idxs, gradients, fraction=args.fraction
)
luscc = LUSCC(las, a_selected, i_selected)
e_luscc, _ = luscc.kernel()
s2 = float(np.real(luscc.s2[0]))
spin = (np.sqrt(1.0 + 4.0*s2) - 1.0) / 2.0
multiplicity = 2.0*spin + 1.0

print(f"LASSCF energy:    {las.e_tot:.12f}")
print(f"LASSIS energy:    {e_lassis[0]:.12f}")
print(f"LAS-LUSCC energy: {e_luscc[0]:.12f}")
print(f"LAS-LUSCC <S^2>:  {s2:.12f}")
print(f"LAS-LUSCC 2S+1:   {multiplicity:.12f}")
print(f"Selected {len(a_selected)} of {len(a_idxs)} excitations")
print("RESULT " + json.dumps({
    "geometry": args.geometry,
    "state": args.state,
    "fraction": args.fraction,
    "nselected": len(a_selected),
    "ntotal": len(a_idxs),
    "energy": float(e_luscc[0]),
    "s2": s2,
    "multiplicity": float(multiplicity),
}, sort_keys=True))
