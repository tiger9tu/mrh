import numpy as np

from pyscf.mcpdft.mcpdft import _PDFT
from pyscf.pbc.dft import gen_grid as pbc_gen_grid
from pyscf.mcpdft._dms import _get_fcisolver

from mrh.my_pyscf.pbc.mcpdft.otfnalperiodic import get_pbc_otfnal_kpts

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

    def _init_ot_grids(self, my_ot, grids_attr=None):
        '''
        Initialization of on-top functional and grids for periodic systems.
        '''
        if grids_attr is None:
            grids_attr = {}

        old_grids = getattr(self, 'grids', None)

        if isinstance(my_ot, (str, np.bytes_)):
            # Note: I have changed the input arg. for below function.
            self.otfnal = get_pbc_otfnal_kpts(self._scf, my_ot)
        else:
            self.otfnal = my_ot

        pbc_grid_types = (
            pbc_gen_grid.UniformGrids,
            pbc_gen_grid.BeckeGrids,
        )

        if isinstance(old_grids, pbc_grid_types):
            self.otfnal.grids = old_grids
        else:
            self.otfnal.grids = pbc_gen_grid.BeckeGrids(self.cell,)

        self.otfnal.grids.__dict__.update(grids_attr)

        for key, value in grids_attr.items():
            assert getattr(self.otfnal.grids, key, None) == value

        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout    
    
    def multi_state(self, method='Lin'):
        raise NotImplementedError(f"StateAverageMix not available for {method}")


    # Need to redefine the casdm1s and casdm2 because of shape mismatch.
    def make_one_casdm1s (self, ci, state=0):
        nkpts = self.nkpts
        ncastot = self.ncas *  nkpts
        fcisolver, ci, nelecas = _get_fcisolver (self, ci, state=state)
        nelecastot = (nelecas[0]*nkpts, nelecas[1]*nkpts)
        return fcisolver.make_rdm1s (ci, ncastot, nelecastot)

    def make_one_casdm2 (self, ci, state=0):
        ncas = self.ncas
        fcisolver, ci, nelecas = _get_fcisolver (self, ci, state=state)
        ncastot = ncas * self.nkpts
        nelecastot = (nelecas[0]*self.nkpts, nelecas[1]*self.nkpts)
        try:
            casdm2 = fcisolver.make_rdm2 (ci, ncastot, nelecastot)
        except AttributeError:
            _, casdm2 = fcisolver.make_rdm12 (ci, ncastot, nelecastot)
        return casdm2


def get_mcpdft_child_class(kmc, ot, **kwargs):
    mc_doc = (kmc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(_kMCPDFT, kmc.__class__):
        __doc__ = mc_doc + '\n\n' + _kMCPDFT.__doc__
        _mc_class = kmc.__class__

        # MC-PDFT object requires mol object in ot.reset functions
        _mc_class.mol = kmc._scf.cell.to_mol()

        def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                                 grids_level=None, grids_attr=None, dump_chk=False, **kwargs):

            '''
            TODO: Make sure the ot and underlying numint are compatible with periodic systems.
            '''
            assert dump_chk is False, "dump_chk is not supported for k-MC-PDFT"
            return _kMCPDFT.compute_pdft_energy_(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                    grids_level=grids_level, grids_attr=grids_attr, dump_chk=False, **kwargs)
     
    pdft = PDFT(kmc._scf, kmc.ncas, kmc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(kmc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft
