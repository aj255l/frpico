from __future__ import annotations

from collections.abc   import Iterable
from functools         import wraps

from frplib_pico.exceptions import ComplexExpectationWarning
from frplib_pico.frps       import FRP, ConditionalFRP
from frplib_pico.kinds      import Kind, ConditionalKind
from frplib_pico.output     import in_panel
from frplib_pico.protocols  import SupportsExpectation, SupportsApproxExpectation, SupportsForcedExpectation
from frplib_pico.quantity   import show_qtuple
from frplib_pico.statistics import Statistic
from frplib_pico.vec_tuples import VecTuple, as_vec_tuple


class Expectation(VecTuple):
    def __init__(self, contents: Iterable):
        self.label = ''

    def __str__(self) -> str:
        return show_qtuple(self)

    def __frplib_repr__(self):
        return in_panel(str(self), title=self.label or None)

def E(x, force_kind=False, allow_approx=True, tolerance=0.01):
    """Computes and returns the expectation of a given object.

    If `x` is an FRP or kind, its expectation is computed directly,
    unless doing so seems computationally inadvisable. In this case,
    the expectation is forced if `forced_kind` is True and otherwise
    is approximated, with specified `tolerance`, if `allow_approx`
    is True.

    In this case, returns a quantity wrapping the expectation that
    allows convenient display at the repl; the actual value is in
    the .this property of the returned object.

    If `x` is a ConditionalKind or ConditionalFRP, then returns
    a *function* from domain values in the conditional to
    expectations.

    """
    if isinstance(x, (ConditionalKind, ConditionalFRP)):
        f = x.expectation()

        @wraps(f)
        def c_expectation(*xs):
            return Expectation(as_vec_tuple(f(*xs)))
        if getattr(f, 'dim') is not None:
            label = f'dimension {getattr(f, "dim")} values'
        else:
            label = 'values'
        setattr(c_expectation, '__frplib_repr__',
                lambda: f'A conditional expectation as a function of {label}.')
        return c_expectation

    if isinstance(x, SupportsExpectation):
        label = ''
        try:
            expect = x.expectation()
        except ComplexExpectationWarning as e:
            if force_kind and isinstance(x, SupportsForcedExpectation):
                expect = x.forced_expectation()
            elif isinstance(x, SupportsApproxExpectation):
                expect = x.approximate_expectation(tolerance)
                label = (f'Computing approximation (tolerance {tolerance}) '
                         f'as exact calculation may be costly')  # ': {str(e)}\n'
            else:
                raise e
        expect = Expectation(as_vec_tuple(expect))
        if label:
            expect.label = label
        return expect
    return None

def D_(X: FRP | Kind):
    """The distribution operator for an FRP or kind.

    When passed an FRP or kind, this returns a function
    that maps any compatible statistic to the expectation
    of the transformed FRP or kind.

    """
    def probe(psi: Statistic):
        # ATTN: Check compatibility here
        return E(psi(X))
    return probe


#
# Info tags
#

setattr(E, '__info__', 'actions')
setattr(D_, '__info__', 'actions')
