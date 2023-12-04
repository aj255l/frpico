from __future__ import annotations

from frplib_pico.kinds      import Kind
from frplib_pico.symbolic   import Symbolic
from frplib_pico.vec_tuples import VecTuple


#
# Kind and FRP based calculations
#

def substitute(quantity, mapping):
    "Substitutes values for symbols in `quantity` from `mapping`."
    if hasattr(quantity, 'this'):  # Facades
        quantity = getattr(quantity, 'this')

    if isinstance(quantity, Symbolic):
        return quantity.substitute(mapping)

    if isinstance(quantity, VecTuple):
        return quantity.map(substitute_with(mapping))

    if isinstance(quantity, Kind):
        f = substitute_with(mapping)
        return quantity.bimap(f, f)

    if isinstance(quantity, list):
        return [substitute(q, mapping) for q in quantity]

    return quantity

def substitute_with(mapping):
    """Returns a function that substitutes values for symbols from `mapping`.

    See `substitute`.

    """
    def sub(quantity, **kw):
        return substitute(quantity, mapping | kw)
    return sub

def substitution(quantity, **kw):
    """Substitutes values for symbols in quantity, with mapping from keywords.

    """
    return substitute(quantity, kw)


#
# Info tags
#

setattr(substitute, '__info__', 'utilities::symbols')
setattr(substitute_with, '__info__', 'utilities::symbols')
setattr(substitution, '__info__', 'utilities::symbols')
