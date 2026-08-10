import numpy as np

from mrh.exploratory.luscc.excitations import (
    apply_operator_string_fci,
    generate_interfragment_excitations,
)


def test_h4_excitation_space_is_stable():
    a_idxs, i_idxs = generate_interfragment_excitations(4, (2, 2))
    assert len(a_idxs) == len(i_idxs) == 146
    assert all(len(a) == len(i) for a, i in zip(a_idxs, i_idxs))
    assert all(sum(np.asarray(a) // 4) == sum(np.asarray(i) // 4)
               for a, i in zip(a_idxs, i_idxs))


def test_operator_application_updates_particle_number():
    ci = np.array([[1.0], [0.0]])
    result, nelec = apply_operator_string_fci(
        ci, norb=2, neleca_nelecb=(1, 0),
        ops=(("ann", 0, "alpha"), ("cre", 1, "alpha")),
    )
    assert nelec == (1, 0)
    assert np.isclose(np.linalg.norm(result), 1.0)
