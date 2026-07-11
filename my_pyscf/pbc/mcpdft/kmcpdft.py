import numpy as np

from pyscf import ao2mo, lib
from pyscf.mcscf.addons import StateAverageMCSCFSolver

from pyscf.mcpdft.otfnal import transfnal, get_transfnal
from pyscf.mcpdft.mcpdft import _get_e_decomp
from pyscf.mcpdft.mcpdft import _PDFT, _mcscf_env

'''
Author: Bhavnesh Jangid
k-MC-PDFT for periodic systems at the gamma point or k-points.
'''

class _kMCPDFT(_PDFT):
    '''
    k-MC-PDFT for periodic systems at the gamma point or k-points.
    This class is making sure, the functionalities which are not 
    compatible with periodic systems are throwing NotImplementedError. 
    '''

    def get_h2eff(self, mo_coeff=None):
        'Compute the active space two-particle Hamiltonian.'
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:, ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:, ncore:nocc]

        if getattr(self._scf, '_eri', None) is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        elif getattr (self, 'with_df', False):
            eri = self.with_df.ao2mo(mo_coeff)

        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

    def multi_state(self, method='Lin'):
        raise NotImplementedError(f"StateAverageMix not available for {method}")


def get_mcpdft_child_class(mc, ot, **kwargs):
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(_kMCPDFT, mc.__class__):
        __doc__ = mc_doc + '\n\n' + _kMCPDFT.__doc__
        _mc_class = mc.__class__
        def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                                 grids_level=None, grids_attr=None, dump_chk=False, **kwargs):

            '''
            TODO: Make sure the ot and underlying numint are compatible with periodic systems.
            '''
            assert dump_chk is False, "dump_chk is not supported for k-MC-PDFT"
            return _kMCPDFT.compute_pdft_energy_(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                    grids_level=grids_level, grids_attr=grids_attr, dump_chk=False, **kwargs)
     
    pdft = PDFT(mc._scf, mc.ncas_sub, mc.nelecas_sub, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft
