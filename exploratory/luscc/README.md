# Experimental LAS-LUSCC

This package implements linearized unitary coupled cluster in a LASSI model
space. It accepts either a LAS reference or a post-kernel LASSI reference and
supports raw and internally contracted solution backends.

```python
from mrh.exploratory.luscc import LUSCC, get_grad_exact

gradients, selected, a_idxs, i_idxs = get_grad_exact(las)
energy, si = LUSCC(las, a_idxs, i_idxs).kernel()
```

The implementation is under `exploratory` while its interfaces and scaling
strategy remain active research topics. General LASSI capabilities required by
the method, including spin-resolved 3-RDMs and spin-targeted Davidson solving,
remain in `mrh.my_pyscf.lassi`.

The excitation generator is implemented locally and does not depend on the
experimental LAS-USCC solver.
