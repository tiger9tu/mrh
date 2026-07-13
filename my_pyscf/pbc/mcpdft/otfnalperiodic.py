
import numpy as np
from functools import reduce

from pyscf.lib import logger, param
from pyscf.mcpdft import _dms
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf.mcpdft.otfnal import otfnal
from pyscf.mcpdft.otfnal import get_transfnal
from pyscf.mcpdft.otfnal import transfnal, ftransfnal
from pyscf import __config__
from pyscf.pbc import gto as pbcgto
from pyscf.mcscf import mc1step, casci
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc.mcscf import mc1step as pbc_mc1step, casci as pbc_casci
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R
from mrh.my_pyscf.pbc.mcscf.mc1step import _get_casdm2_kpts as _basis_transform_casdm2_kpts
from pyscf import gto
from pyscf.pbc import dft

def redefine_fnal(original_class, new_parent):
    from pyscf import lib
    class transfnal (original_class.__class__, new_parent):
        pass
    new_fnal = lib.view (original_class, transfnal)
    return new_fnal

redefine_transfnal = redefine_fnal
redefine_ftransfnal = redefine_fnal

def _get_mol_or_cell(kmc_or_kmf_mol_cell):
    '''
    A function to get the mol object from the kmc_or_kmf_mol object
    '''
    if isinstance(kmc_or_kmf_mol_cell, (mc1step.CASSCF, casci.CASCI)):
        return kmc_or_kmf_mol_cell._scf.mol
    elif isinstance(kmc_or_kmf_mol_cell, (pbc_mc1step.CASSCF, pbc_casci.CASCI)):
        return kmc_or_kmf_mol_cell._scf.cell
    elif isinstance(kmc_or_kmf_mol_cell, gto.Mole) or \
        isinstance(kmc_or_kmf_mol_cell, pbcgto.cell.Cell):
        return kmc_or_kmf_mol_cell
    elif getattr(kmc_or_kmf_mol_cell, 'mol', None) is not None:
        return kmc_or_kmf_mol_cell.mol
    elif getattr(kmc_or_kmf_mol_cell, 'cell', None) is not None:
        return kmc_or_kmf_mol_cell.cell
    else:
        raise ValueError ("The input object is not recognized. " \
        "It should be either MC-SCF/SCF or Mole/Cell object.")

class otfnalperiodic(otfnal):
    '''
    Child class to define the otfnal class for periodic systems (Only for 1x1x1 kpts)
    '''

    def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=param.MAX_MEMORY, hermi=1):
        '''
        See the docstring of pyscf/mcpdft/otfnal.energy_ot for more information.
        '''

        E_ot = 0.0
        ni = ot._numint
        xctype =  ot.xctype

        if xctype=='HF': 
            return E_ot
        
        dens_deriv = ot.dens_deriv

        nao = mo_coeff.shape[0]
        ncas = casdm2.shape[0]
        cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
        
        dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                    ncas=ncas)
        mo_cas = mo_coeff[:,ncore:][:,:ncas]
        t0 = (logger.process_clock (), logger.perf_counter ())
        make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi) for
            i in range(2))
        
        for ao_k1, ao_k2, mask, weight, _ \
            in ni.block_loop(ot.mol, ot.grids, nao, deriv=dens_deriv, kpt=None, max_memory=max_memory):
            '''
            ao_k1 and ao_k2 are the block of AO integrals for the given k-point. They
            are the same for supercell(1x1x1) calculations.
            '''

            rho = np.asarray ([m[0] (0, ao_k1, mask, xctype) for m in make_rho])
            t0 = logger.timer (ot, 'untransformed density', *t0)
            Pi = get_ontop_pair_density (ot, rho, ao_k1, cascm2, mo_cas,
                dens_deriv, mask)
            t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
            if rho.ndim == 2:
                rho = np.expand_dims (rho, 1)
                Pi = np.expand_dims (Pi, 0)
            E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
            t0 = logger.timer (ot, 'on-top energy calculation', *t0)

        return E_ot
    
    energy_ot.__doc__ = otfnal.energy_ot.__doc__

    def reset(self, mol=None):
        '''
        Discard cached grid data and optionally update the cell object.
        I am not changing the input parameter so that it is compatible with the current
        MCPDFT code.
        '''
        if mol is not None:
            self.mol = mol
        # A hack to reset the grids for the new cell object.
        self.grids.reset (mol) 

otfnalperiodic_gamma = otfnalperiodic

class otfnalperiodic_kpts(otfnal):
    '''
    Child class to define the otfnal class for periodic systems for k-points.
    '''

    def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=param.MAX_MEMORY, hermi=1):
        '''
        See the docstring of pyscf/mcpdft/otfnal.energy_ot for more information.
        '''

        E_ot = 0.0
        ni = ot._numint
        xctype =  ot.xctype
        dtype = mo_coeff.dtype

        if xctype=='HF': 
            return E_ot
        
        assert mo_coeff.ndim == 3, "The mo_coeff should be 3D array for k-points calculations"
        
        dens_deriv = ot.dens_deriv

        nao = mo_coeff[0].shape[0]
        ncastot = casdm2.shape[0]
        ncas = ot.ncas
        nkpts = ncastot // ncas
        
        # Need to check this.
        cell = ot.cell
        kpts = ot.kpts

        assert nkpts == mo_coeff.shape[0], "The number of k-points in mo_coeff and casdm2 should be same"
        assert getattr(ot, 'kmesh', None) is not None, "The kmesh attribute should be set in the otfnal object"

        mo_phase = get_mo_coeff_k2R(ot._scf, mo_coeff, ncore, ncas, kmesh=ot.kmesh)[-1]
        
        assert casdm2.shape == (ncastot,)*4
        assert casdm1s.shape == (ncastot, ncastot)
        
        # First construct the cumulant then transform it to block mo-orbitals basis.
        cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
        casdm2_kpts = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=dtype)
        kconserv = kpts_helper.get_kconserv(cell, kpts)
        for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
            k4 = kconserv[k1, k2, k3]
            dm2_k = _basis_transform_casdm2_kpts(cascm2, mo_phase, (k1, k2, k3, k4))
            casdm2_kpts[k1, k2, k3] = dm2_k
        
        # First, transform the casdm1s to dm1s for each k-point.
        dm1s_kpts = []
        for k in range(nkpts):
            casdm1s_k = [reduce(np.dot, (mo_phase[k], casdm1s_, mo_phase[k].conj().T)) 
                        for casdm1s_ in casdm1s]
            dm1s =_dms.casdm1s_to_dm1s (ot, casdm1s_k, mo_coeff=mo_coeff[k], ncore=ncore, 
                                         ncas=ncas)
            dm1s_kpts.append(dm1s)
        
        dm1s_kpts = np.array(dm1s_kpts)

        mo_cas = np.array([mo_coeff[k][:,ncore:][:,:ncas] 
                           for k in range(nkpts)])
        
        t0 = (logger.process_clock (), logger.perf_counter ())
        make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi) for
            i in range(2))
        
        for ao_k1, ao_k2, mask, weight, _ \
            in ni.block_loop(ot.mol, ot.grids, nao, deriv=dens_deriv, kpt=None, max_memory=max_memory):
            '''
            ao_k1 and ao_k2 are the block of AO integrals for the given k-point. They
            are the same for supercell(1x1x1) calculations.
            '''
            rho = np.asarray ([m[0] (0, ao_k1, mask, xctype) for m in make_rho])
            t0 = logger.timer (ot, 'untransformed density', *t0)
            Pi = get_ontop_pair_density (ot, rho, ao_k1, cascm2, mo_cas,
                dens_deriv, mask)
            t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
            if rho.ndim == 2:
                rho = np.expand_dims (rho, 1)
                Pi = np.expand_dims (Pi, 0)
            E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
            t0 = logger.timer (ot, 'on-top energy calculation', *t0)

        return E_ot
    
    energy_ot.__doc__ = otfnal.energy_ot.__doc__



def _get_ks_obj(kmc_or_kmf_or_cell):
    '''
    Initialize KS object with app. density fitting object GDF, MDF or FFTDF
    args:
        kmc_or_kmf_or_cell : kMC or kMF object with cell object
    returns:
        ks : KS object with app. density fitting object GDF, MDF or FFTDF
    '''
    cell = _get_mol_or_cell (kmc_or_kmf_or_cell)
    if hasattr(kmc_or_kmf_or_cell, 'with_df'):
        dfclass = kmc_or_kmf_or_cell.with_df.__class__.__name__
    
    elif hasattr(kmc_or_kmf_or_cell, '_las'):
        dfclass = kmc_or_kmf_or_cell._las.with_df.__class__.__name__
    else:
        raise ValueError ("The input object does not have with_df attribute. \
                          Start with Mean-field object")
    if dfclass == 'GDF':
        ks = dft.RKS(cell).density_fit()
    elif dfclass == 'MDF':
        ks = dft.RKS(cell).mix_density_fit()
    else:
        raise NotImplementedError ("PBD-MCPDFT is yet not implemented for FFTDF")
    return ks

def get_pbc_otfnal(kmc_or_kmf_or_cell, otxc):
    '''
    This is wrapper function to get the appropriate fnal class 
    for the given cell object
    args:
        kmc_or_kmf_or_cell : kMC or kMF object with cell object
        otxc : str, on-top functional name
    '''
    cell = _get_mol_or_cell (kmc_or_kmf_or_cell)
    fnal_class = get_transfnal (cell, otxc)
    fnal_class_type = fnal_class.__class__.__name__

    assert isinstance(otxc, str), "The otxc should be a string"
    xc_base = fnal_class.otxc

    ks = _get_ks_obj(kmc_or_kmf_or_cell)
    
    if fnal_class_type == 'transfnal':
        xc_base = xc_base[1:]
        ks.xc = xc_base
        org_transfnal = transfnal(ks)
        new_func_class = redefine_transfnal (org_transfnal, otfnalperiodic_gamma)
        del org_transfnal

    elif fnal_class_type == 'ftransfnal':
        xc_base = xc_base[2:]
        ks.xc = xc_base
        org_ftransfnal = ftransfnal(ks)
        new_func_class = redefine_ftransfnal (org_ftransfnal, otfnalperiodic_gamma)
        del org_ftransfnal
    else:
        raise ValueError ("The fnal class is not recognized")

    logger.info(cell, 'Periodic OT-FNAL class is used')
    return new_func_class
