from __future__ import annotations

from frplib_pico.exceptions import IndexingError
from frplib_pico.kinds      import Kind
from frplib_pico.frps       import FRP
from frplib_pico.statistics import Proj


#
# Utilities
#

def components(frp_or_kind: Kind | FRP):
    """Returns the components of an FRP or Kind.

    If X_ = components(X) for an FRP/Kind X, then X_[i] is the ith
    component FRP/Kind. Here, i must be between 1 and dim(X).
    This can also be used as an iterator:

        for x in components(X):
            ...

    In most cases, it is easier to subscript the FRP or Kind
    directly, like X[i].

    """
    class Component:
        def __init__(self, frp_or_kind: Kind | FRP, dim: int):
            self.this = frp_or_kind
            self.dim = dim

        def __getitem__(self, index):
            if index < 1 or index > self.dim:
                raise IndexingError(
                    f'The {index} component of this FRP/Kind does not exist; its dimension is {self.dim}.'
                )
            return Proj[self.dim](self.this)

        def __iter__(self):
            for i in range(1, self.dim + 1):
                yield self[i]

    return Component(frp_or_kind, frp_or_kind.dim)
