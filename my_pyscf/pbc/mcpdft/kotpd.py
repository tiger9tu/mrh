import numpy as np

def get_ontop_pair_density_kpts(ot, rho, ao, cascm2, mo_cas,
        kconserv, deriv=0, non0tab=None):
    r'''
    Compute the on-top pair density for a k-point sampled
    multiconfigurational wave function:
        Pi(r) = rho_alpha(r) * rho_beta(r) + Lambda(r) / 2

    Here:
        Lambda(r) = 1 / Nk**2
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
        ao : ndarray (nkpts, *, ngrids, nao)
            Periodic AO values on the numerical grid.
        cascm2 : ndarray (nkpts**3, ncas, ncas, ncas, ncas)
            Spin-summed active-space two-body cumulant.
            The first three k-point indices correspond to
            k1, k2, and k3. The fourth k-point is obtained from
            momentum conservation.
        mo_cas : ndarray of shape (nkpts, nao, ncas)
            Active molecular-orbital coefficients at each k-point.
        kconserv : ndarray of shape (nkpts, nkpts, nkpts)
            k-point conservation table
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

    # Remove the derivative axis when only zeroth-order values
    # are supplied.
    if rho.ndim == 3:
        if rho.shape[1] != 1:
            raise ValueError(
                'For deriv=0, rho must have shape '
                '(2,ngrids) or (2,1,ngrids)'
            )
        rho = rho[:, 0]

    if ao.ndim == 4:
        if ao.shape[1] != 1:
            raise ValueError(
                'For deriv=0, ao must have shape '
                '(nkpts,ngrids,nao) or '
                '(nkpts,1,ngrids,nao)'
            )
        ao = ao[:, 0]

    if rho.ndim != 2 or rho.shape[0] != 2:
        raise ValueError(
            'rho must have shape (2,ngrids)'
        )

    if ao.ndim != 3:
        raise ValueError(
            'ao must have shape (nkpts,ngrids,nao)'
        )

    if mo_cas.ndim != 3:
        raise ValueError(
            'mo_cas must have shape (nkpts,nao,ncas)'
        )

    nkpts, ngrids, nao = ao.shape
    nkpts_mo, nao_mo, ncas = mo_cas.shape

    if nkpts_mo != nkpts:
        raise ValueError(
            'ao and mo_cas contain different numbers of k-points'
        )

    if nao_mo != nao:
        raise ValueError(
            'AO dimension of ao and mo_cas is inconsistent'
        )

    if rho.shape[1] != ngrids:
        raise ValueError(
            'rho and ao contain different numbers of grid points'
        )

    # Evaluate active Bloch MOs on the grid:
    #
    #     phi[k,g,u] = sum_mu ao[k,g,mu] * C[k,mu,u]
    #
    grid2amo = np.einsum(
        'kga,kau->kgu',
        ao,
        mo_cas,
        optimize=True,
    )

    dtype = np.result_type(
        rho.dtype,
        grid2amo.dtype,
        cascm2.dtype,
    )

    # Disconnected contribution. rho is assumed to already be
    # k-point averaged.
    Pi = np.asarray(
        rho[0] * rho[1],
        dtype=dtype,
    ).copy()

    Pi_connected = np.zeros(
        ngrids,
        dtype=dtype,
    )

    for k1 in range(nkpts):
        phi1H = grid2amo[k1].conj()
        for k2 in range(nkpts):
            phi2H = grid2amo[k2].conj()
            gridkern_left = (
                phi1H[:, :, np.newaxis]
                * phi2H[:, np.newaxis, :]
            )
            for k3 in range(nkpts):

                # k-point conservation table:
                #
                #     kconserv[ki,kj,kk] = kl
                #
                # with
                #
                #     ki - kj + kk - kl = G.
                #
                # We require
                #
                #     k1 + k2 - k3 - k4 = G,
                #
                # so use ki=k1, kj=k3, kk=k2.
                k4 = kconserv[k1, k3, k2]

                phi3 = grid2amo[k3]
                phi4 = grid2amo[k4]

                # phi3[g,x] * phi4[g,y]
                #
                # Shape: (ngrids,ncas,ncas)
                gridkern_right = (
                    phi3[:, :, np.newaxis]
                    * phi4[:, np.newaxis, :]
                )

                if cascm2.ndim == 5:
                    # Flattened ordering:
                    #
                    #     ik123 = ((k1 * nkpts) + k2) * nkpts + k3
                    #
                    ik123 = (
                        (k1 * nkpts + k2) * nkpts + k3
                    )
                    cm2_k = cascm2[ik123]

                elif cascm2.ndim == 7:
                    cm2_k = cascm2[k1, k2, k3]

                else:
                    raise ValueError(
                        'cascm2 must have shape '
                        '(nkpts**3,ncas,ncas,ncas,ncas) '
                        'or '
                        '(nkpts,nkpts,nkpts,'
                        'ncas,ncas,ncas,ncas)'
                    )

                # Contract the first two cumulant indices:
                #
                #     wrk[g,x,y]
                #       = sum_{u,v}
                #         phi[u,k1]^*
                #         phi[v,k2]^*
                #         Lambda[u,v,x,y]
                #
                wrk = np.tensordot(
                    gridkern_left,
                    cm2_k,
                    axes=((1, 2), (0, 1)),
                )

                # Complete the x,y contraction:
                #
                #     sum_{x,y} wrk[g,x,y]
                #                 phi[x,k3]
                #                 phi[y,k4]
                #
                Pi_connected += (
                    gridkern_right * wrk
                ).sum(axis=(1, 2))

    Pi += Pi_connected / (2.0 * nkpts**2)

    # The complete k-point sum must be real, although individual
    # k-point contributions are generally complex.
    return np.real_if_close(Pi)