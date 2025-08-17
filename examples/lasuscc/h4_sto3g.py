# Author: Shreya Verma shreyav@uchicago.edu
# This is a sample script to run LAS-USCCSD for the H4 molecule with the polynomial-scaling algorithm to select cluster excitations
# (2e,2o)+(2e,1o)
# This is not a VQE calculation with statevector simulator, rather the classical emulator is used 

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
import time

# Initializing the molecule with RHF
#===================================
xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI


# Running LASSCF
#===================================
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
las.verbose = 4
# las.nroots = 2
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.state_average_(weights=[0.5,0.5], spins=[[1,1],[1,-1]])
las_e_tot, las_e_states, las_ci, las_mo_coeff, las_mo_energy, las_h2eff_sub, las_veff = las.kernel (mo_loc)

print("las e_states = ", las_e_states)
print("las ci [0]= \n", las_ci[0])

# print("las e_tot = ", las_e_tot)
# print("las e_cas = ", las_e_cas)
# print("las mo_coeff = \n", las_mo_coeff)
# print("las mo_energy = \n", las_mo_energy)
# print("las h2eff_sub = \n", las_h2eff_sub)
# print("las veff = \n", las_veff)

#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================
# all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon=0.0001)
# # print ("All gradients = ", all_g)
# # print ("Selected gradients = ", g_sel)

# excitations = []
# for a, i in zip(a_idxs_selected, i_idxs_selected):
#     excitations.append((tuple(i), tuple(a[::-1])))

# print ("Selected excitations = ", excitations)

#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================
# epsilon=0.001
# mc_uscc = mcscf.CASCI(mf, 4, 4)
# mc_uscc.kernel()


# mc_uscc.mo_coeff = las.mo_coeff
# lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
# mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
# mc_uscc.fcisolver.norb_f = [2,2]
# mc_uscc.fcisolver.frozen = 'CI'
# start_time = time.time()
# mc_uscc.kernel()

# print("mc_uscc ci shape = ", mc_uscc.ci.shape)
# end_time = time.time()
# print("Kernel execution time: {:.6f} seconds".format(end_time - start_time))
# print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_selected), mc_uscc.e_tot))
# print("ref energy = ", ref.e_tot)
# print("las energy = ", las.e_tot)