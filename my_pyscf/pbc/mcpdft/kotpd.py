import numpy as np
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper

def get_ontop_pair_density_kpts(ot, rho, ao, cascm2, mo_cas,
        kconserv, deriv=0, non0tab=None):
    r'''
    Compute the on-top pair density for a k-point sampled
    multiconfigurational wave function:
        Pi(r) = rho_alpha(r) * rho_beta(r) + Lambda(r) / 2

    Here:
        Lambda(r) = 1 /(2 *Nk**2)
            * sum_{k1,k2,k3}
            * sum_{u,v,x,y}
                phi_{u,k1}(r)
                phi_{v,k2}(r)
                cascm2[k1,k2,k3,u,v,x,y]
                phi_{x,k3}(r)
                phi_{y,k4}(r)

    and k4 is determined from momentum conservation:

        k1 - k2 + k3 - k4 = G.

    Args:
        ot :
            On-top pair-density functional object.
        rho :  (2, *, ngrids)
            Spin-separated, k-point averaged density.
            Note: rho[0] and rho[1] must already include the scaling factor of
            1 / Nk k-point averaging.
        ao : ndarray (nkpts, [comp], ngrids, nao)
            Periodic AO values on the numerical grid.
            comp can be 1 or 4, and so on. 
        cascm2 : ndarray (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
            Spin-summed active-space two-body cumulant.
            The first three k-point indices correspond to
            k1, k2, and k3. The fourth k-point is obtained from
            momentum conservation.
        mo_cas : ndarray of shape (nkpts, nao, ncas)
            Active molecular-orbital coefficients at each k-point.
        kconserv : ndarray of shape (nkpts, nkpts, nkpts)
            k-point conservation table
    Kwargs:
        deriv : int
            Only deriv=0 is implemented.
        non0tab :
            Included for compatibility. Not used in this direct
            implementation.
    Returns:
        Pi : ndarray of shape (ngrids,)
            On-top pair density.
    '''
    assert deriv == 0, 'Only zeroth-order on-top pair density is implemented'

    rho = np.asarray(rho)
    ao = np.asarray(ao)
    cascm2 = np.asarray(cascm2)
    mo_cas = np.asarray(mo_cas)
    kconserv = np.asarray(kconserv)

    # Some sanity checks:
    assert mo_cas.shape[0] == ao.shape[0] == cascm2.shape[0]
    assert rho.ndim == 3 and rho.shape[0] == 2, 'rho must have shape (2,*,ngrids)'
    assert ao.ndim == 4 and ao.shape[1] == 1, 'ao must have shape (nkpts,*,ngrids,nao)'

    dtype = np.result_type(rho.dtype, grid2amo.dtype, cascm2.dtype,)

    nkpts = mo_cas.shape[0]
    ngrids, nao = ao.shape[-2:]

    # Evaluate active Bloch MOs on the grid:
    #     phi[k,g,u] = sum_mu ao[k,g,mu] * C[k,mu,u]
    t0 = (logger.process_clock (), logger.perf_counter ())

    grid2amo = np.array([np.einsum('kga,kau->kgu', ao[k, 0], mo_cas[k], optimize=True)
                for k in range(nkpts)], dtype=dtype)

    Pi_shape = ((1,4,5)[deriv], rho.shape[-1])
    Pi = np.zeros(Pi_shape, dtype=dtype)

    # Disconnected contribution. Note that, rho is assumed to already be
    # k-point averaged.
    # Pi = rho_alpha * rho_beta
    Pi += rho[0, 0] * rho[1, 0]

    # Time for the connected parts:
    Pi_connected = np.zeros(ngrids, dtype=dtype,)

    # for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
    #     k4 = kconserv[k1, k2, k3]
    #     # phi1[g,x] * phi2[g,y] : result shape (ngrids, ncas, ncas)
    #     phi1H = grid2amo[k1].conj()
    #     phi2H = grid2amo[k2]
    #     gridkern_left = (phi1H[:, :, None] * phi2H[:, None, :])
    #     # phi13g,x] * phi4[g,y] : result shape (ngrids, ncas, ncas)
    #     phi3 = grid2amo[k3].conj()
    #     phi4 = grid2amo[k4]
    #     gridkern_right = (phi3[:, :, None] * phi4[:, None, :])
    #     cm2_k = cascm2[k1, k2, k3]

    #     # wrk[g,x,y] = sum_{u,v} phi[u,k1]^* * phi[v,k2] * Lambda[u,v,x,y]
    #     wrk = np.tensordot(gridkern_left, cm2_k, axes=((1, 2), (0, 1)),)

    #     #     = sum_{x,y} wrk[g,x,y] * phi[x,k3]^* * phi[y,k4]
    #     Pi_connected += (gridkern_right * wrk).sum(axis=(1, 2))

    for k1 in range(nkpts):
        phi1H = grid2amo[k1].conj()
        for k2 in range(nkpts):
            phi2H = grid2amo[k2]
            gridkern_left = (phi1H[:, :, None] * phi2H[:, None, :])
            cm2_k = cascm2[k1, k2, k3]
            # wrk[g,x,y] = sum_{u,v} phi[u,k1]^* * phi[v,k2] * Lambda[u,v,x,y]
            wrk = np.tensordot(gridkern_left, cm2_k, axes=((1, 2), (0, 1)),)
            for k3 in range(nkpts):
                k4 = kconserv[k1, k2, k3]
                phi3 = grid2amo[k3].conj()
                phi4 = grid2amo[k4]
                gridkern_right = (phi3[:, :, None] * phi4[:, None, :])
                #     = sum_{x,y} wrk[g,x,y] * phi[x,k3]^* * phi[y,k4]
                Pi_connected += (gridkern_right * wrk).sum(axis=(1, 2))

    # Don't forget to normalize it by number of k-points.
    Pi += Pi_connected / (2.0 * nkpts**2)
    t0 = logger.timer_debug1 (ot, 'otpd takes: ', *t0)
    return Pi

