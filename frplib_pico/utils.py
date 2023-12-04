from __future__ import annotations

import sys

from collections       import defaultdict
from collections.abc   import Iterable, Hashable
from functools         import reduce
from typing            import Callable, Generator, TypeVar, Union
from typing_extensions import Any, TypeGuard

from frplib_pico.env        import environment
from frplib_pico.exceptions import ConstructionError, OperationError
from frplib_pico.protocols  import Renderable

#
# Generic
#

A = TypeVar('A')

def identity(x: A) -> A:
    "Returns its argument."
    return x

def const(a: A) -> Callable[[Any], A]:
    "Returns a constant function that returns the given value."
    def const_fn(x: Any) -> A:
        return a
    return const_fn


#
# Kinds and FRPs and Such
#

def values(x, scalarize=False) -> set:
    """Returns the set of values of a kind.

    Parameters:
    ----------
      x - a kind, or any object with a .values property
      scalarize [False] - if True, convert numeric scalars to floats

    Returns a set of possible values, with scalars unwrapped from
    their tuples.

    """
    try:
        if scalarize:
            return set(map(float, x.values))
        return x.values
    except Exception:
        raise OperationError(f'Object {str(x)} does not have a values property.')

def dim(x):
    "Returns the dimension of its argument, which is typically a kind, FRP, or statistic."
    try:
        return x.dim
    except Exception:
        raise OperationError(f'Object {str(x)} does not have a dim property.')

def codim(x):
    "Returns the co-dimension of its argument, which is typically a kind, FRP, or statistic."
    try:
        return x.codim
    except Exception:
        raise OperationError(f'Object {str(x)} does not have a codim property.')

def size(x):
    "Returns the size of its argument, which is typically a kind or FRP."
    try:
        return x.size
    except Exception:
        raise OperationError(f'Object {str(x)} does not have a size property.')

def clone(x):
    """Returns a clone of its argument, which is typically an FRP or conditional FRP.

    This operates on any object with a .clone() method, but is typically used
    to get a copy of an FRP or conditional FRP with the same kind but its
    own value.

    """
    try:
        return x.clone()
    except Exception as e:
        raise OperationError(f'Could not clone object {x}: {str(e)}')


#
# Tuples
#

def is_tuple(x: Any) -> TypeGuard[tuple]:
    "Is the given object a tuple?"
    return isinstance(x, tuple)

def scalarize(x):
    "If given a length 1 tuple, unwrap the value; otherwise returns argument as is."
    return (x[0] if isinstance(x, tuple) and len(x) == 1 else x)

def ensure_tuple(x: Any) -> tuple:
    "If given a non-tuple, wrap in a length 1 tuple; else returns argument as is."
    return (x if isinstance(x, tuple) else (x,))


#
# Sequences and Collections
#

def irange(
        start_or_stop: int,
        stop: int | None = None,
        *,
        step=1,
        exclude: Callable[[int], bool] | Iterable[int] | None = None,
        include: Callable[[int], bool] | Iterable[int] | None = None,
) -> Generator[int, None, None]:
    """Inclusive integer range.

    Parameters
    ----------
      start_or_stop - if the only argument, an integer giving the stop (inclusive)
          of the sequence; if stop is also supplied, this is the start.
      stop - if missing, start from 1 (unlike the builtin range that starts from 0);
          otherwise, the sequence goes up to and including this value.
      step - a non-zero integer giving the spacing between successive values of the
          sequence; it can e negativ if stop < start.
      exclude - either a set of integers or a predicate taking integers to boolean
          values; values in the set or for which the predicate returns true are skippe.
      include - either a set of integers or a predicate taking integers to boolean
          values; values in the set or for which the predicate returns true are included.
          If exclude is also supplied, this takes precedence.

    Returns an iterator over the resulting range.

    """
    if exclude is not None and not callable(exclude):
        exclude_values = set(exclude)
        exclude = lambda x: x in exclude_values
    if include is not None and not callable(include):
        include_values = set(include)
        include = lambda x: x in include_values

    if stop is None:
        stop = start_or_stop
        start = 1
    else:
        start = start_or_stop

    if (stop - start) * step < 0:
        raise ConstructionError(f'irange {start}:{stop} and step {step} have inconsistent direction.')

    sign = 1 if step >= 0 else -1

    def generate_from_irange() -> Generator[int, None, None]:
        value = start
        while (value - stop) * sign <= 0:
            if ((include is None and exclude is None) or
               (include is not None and include(value)) or
               (exclude is not None and not exclude(value))):
                yield value
            value += step

    return generate_from_irange()

def index_of(value, xs, not_found=-1, *, start=0, stop=sys.maxsize):
    """Returns index of `value` in `xs`, or `not_found` if none.

    If xs is a list or tuple, restrict attention to the slice
    from start to stop, exclusive, where start <= stop.

    """
    if stop <= start:
        return not_found

    if isinstance(xs, (list, tuple)):
        try:
            return xs.index(value, start, stop)
        except ValueError:
            return not_found
    else:
        for i, v in enumerate(xs):
            if i >= start and i < stop and v == value:
                return i
        return not_found

def frequencies(xs: Iterable[Hashable], counts_only=False) -> Union[dict[Hashable, int], tuple[int, ...]]:
    """Computes frequencies of the values in some iterable collection.

    If counts_only is False, returns a dictionary mapping the values to their counts.
    Otherwise, returns a tuple of counts in decreasing order.

    The items in the collection should be hashable.

    """
    freqs: dict[Hashable, int] = defaultdict(int)

    for x in xs:
        freqs[x] += 1

    if counts_only:
        return tuple(sorted(freqs.values(), reverse=True))
    return freqs


#
# Higher-Order Functions
#

def compose(*functions):
    """Returns a new function that composes the given functions successively.

    Note that compose(f,g) calls f *after* g. The values of g should be
    valid inputs to f, and similarly for any list of functions.
    This is not checked, however.

    """
    def compose2(f, g):
        return lambda x: f(g(x))

    n = len(functions)
    if n == 0:
        return identity

    if n == 1:
        return functions[0]

    if n == 2:
        return compose2(functions[0], functions[1])

    return reduce(compose2, functions)

    #  # For later Python versions
    # match functions:
    #     case ():
    #         return identity
    #     case (f,):
    #         return f
    #     case (f, g):
    #         return compose2(f, g)
    #     case _:
    #         return reduce(compose2, functions )

def lmap(func, *iterables):
    "Like the builtin `map` but automatically converts its results into a list."
    return list(map(func, *iterables))

def every(func, iterable):
    "Returns true if f(x) is truthy for every x in iterable."
    return all(map(func, iterable))

def some(func, iterable):
    "Returns true if f(x) is truthy for some x in iterable."
    return any(map(func, iterable))

def iterate(f: Callable[..., A], n: int, start: A, *extra_args, **extra_kwargs) -> A:
    """Returns nth item in the sequence: start, f(start), f(f(start)), f(f(f(start))), ...

    If n <= 0, start is returned as is. Extra positional and keyword arguments
    are passed to f in each call.

    Parameters
    ----------
    f :: a function from A -> A, however f can accept extra positional or keyword
        arguments; these are given by `extra_args` and `extra_kwargs`, respectively.

    n :: the number of times to iterate; if n <= 0, `start` is returned as is.

    start :: a value of type A, the initial value of the sequence

    extra_args :: zero or more additional arguments that are passed to `f`
        following the value of type A.

    extra_kwargs :: zero or more additional keyword arguments that are passed
        to `f` following the value of type A and `extra_args`.

    """
    result = start
    for _ in range(n):
        result = f(result, *extra_args, **extra_kwargs)
    return result


#
# Environment
#

def is_interactive() -> bool:
    "Checks if frp is running as an interactive app."
    return environment.is_interactive or hasattr(sys, 'ps1') or bool(sys.flags.interactive)

# ATTN: This needs work but is still useful
def show(x, *, print_it=True, indent=0, render=True):
    "Shows nested objects in the REPL in a more presentable fashion."
    if render and isinstance(x, Renderable):
        out = x.__frplib_repr__()
    elif isinstance(x, list):
        ind0 = (" " * indent)
        ind = ind0 + "  "
        sep = "\n" + ind
        init = "[\n" + ind
        final = "\n" + ind0 + "]"
        # Note: handle panels and block text
        out = init + sep.join([show(xi, print_it=False, indent=indent + 2, render=False)
                               for xi in x]) + final
    elif isinstance(x, dict):
        ind0 = (" " * indent)
        out = str({k: show(v, print_it=False) for k, v in x.items()})
    else:
        ind0 = (" " * indent)
        out = ind0 + str(x)
    if print_it:
        environment.console.print(out)
        return
    return out


#
# Info tags
#

setattr(clone, '__info__', 'utilities')
setattr(dim, '__info__', 'utilities')
setattr(codim, '__info__', 'utilities')
setattr(size, '__info__', 'utilities')
setattr(values, '__info__', 'utilities')
setattr(identity, '__info__', 'utilities')
setattr(const, '__info__', 'utilities')
setattr(compose, '__info__', 'utilities')
setattr(irange, '__info__', 'utilities::irange')
setattr(index_of, '__info__', 'utilities::index_of')
setattr(every, '__info__', 'utilities')
setattr(some, '__info__', 'utilities')
setattr(lmap, '__info__', 'utilities')
setattr(is_tuple, '__info__', 'utilities')
setattr(frequencies, '__info__', 'utilities')
setattr(show, '__info__', 'utilities::show')
setattr(iterate, '__info__', 'utilities::iterate')
