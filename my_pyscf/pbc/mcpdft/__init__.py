#!/bin/bash

import copy
import numpy as np

from pyscf.pbc import scf, dft

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcpdft.kmcpdft import get_mcpdft_child_class

# Author: Bhavnesh Jangid
# Implementing k-MC-PDFT. For initialization, I am using different function, (as sanity checks will be different.)
# However, I will try to import as much code from molecular PDFT and same code structure.

def _sanity_check_for_kmf(kmf0):
    '''
    Wrapper function to check whether the input mean-field object is periodic SCF or not.
    If it is k-DFT then convert that to the k-HF object.
    '''
    assert isinstance(kmf0, scf.hf.SCF),  \
        "k-MCPDFT only works with periodic SCF objects"

    if isinstance(kmf0, dft.krks.KRKS) or isinstance(kmf0, dft.kuks.KUKS) \
        or isinstance(kmf0, dft.rks.RKS) or isinstance(kmf0, dft.uks.UKS):
        raise NotImplementedError("k-MCPDFT with DFT as the base method is not implemented yet.")
    
    if isinstance(kmf0, scf.kuhf.KUHF):
        kmf0 = scf.addons.convert_to_rhf(kmf0)
    
    return kmf0


def _MCPDFT (mc_class, kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None,
             **kwargs):

    kmf0 = getattr (kmc_or_kmf, '_scf', None)
    
    # If started with kCASCI or kCASSCF object, 
    if kmf0 is not None:
        kmf0 = _sanity_check_for_kmf(kmf0)
        kmc0 = kmc_or_kmf
    else:
        kmf0 = kmc_or_kmf
        kmf0 = _sanity_check_for_kmf(kmf0)
        kmc0 = None

    assert frozen is None, "Frozen orbitals are not supported in k-MCPDFT yet."
    kmc = get_mcpdft_child_class (mc_class (kmf0, ncas, nelecas, ncore=ncore),
                                   ot, **kwargs)

    if kmc0 is not None:
        kmc.kmesh = kmc0.kmesh
        kmc.kpts = kmc0.kpts
        kmc.verbose = kmc0.verbose
        kmc.stdout = kmc0.stdout
        kmc.mo_coeff = kmc_or_kmf.mo_coeff.copy()
        kmc.ci = copy.deepcopy (kmc_or_kmf.ci)
        kmc.converged = kmc0.converged
    return kmc

def kCASSCFPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    return _MCPDFT(mcscf.CASSCF, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                   **kwargs)

def kCASCIPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    return _MCPDFT(mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                   **kwargs)

CASSCF = kCASSCFPDFT
CASCI = kCASCIPDFT
