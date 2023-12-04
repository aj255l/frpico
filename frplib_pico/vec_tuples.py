from __future__ import annotations

import math

from collections.abc   import Iterable
from decimal           import Decimal
from fractions         import Fraction
from functools         import reduce
from operator          import add, mul, sub
from typing            import cast, Type, TypeVar, Union
from typing_extensions import Self, TypeGuard

from frplib_pico.exceptions import OperationError, NumericConversionError
from frplib_pico.numeric    import Numeric, NumericF, NumericD, NumericB, numeric_sqrt  # ATTN: Numeric+Symbolic+SupportsVec
from frplib_pico.numeric    import as_numeric as scalar_as_numeric
from frplib_pico.symbolic   import Symbolic, symbolic_sqrt

# SupportsVec  mixin can allow Symbolic and VecTuple automatically,   __plus__  __scalar_mul__
# SupportsNumeric protocol __numeric__ with numeric conversion.
# Can newtype Decimal and Fraction etc.; subclass with __slots__

#
# Types
#

# A VecTuple should contain entirely interoperable types
T = TypeVar('T', NumericF, NumericD, NumericB, Union[Numeric, Symbolic])


#
# Helpers
#

def extend(x, k, scalar_only=False):
    # Want T here but allow Quantity here when available
    if isinstance(x, (int, float, Fraction, Decimal, Symbolic)):
        return VecTuple([x] * k)
    if not scalar_only and isinstance(x, (tuple, list)) and len(x) == 1:
        return VecTuple([*x] * k)
    return x

def from_scalar(x):
    # Want T here but allow Quantity here when available
    if isinstance(x, (int, float, Fraction, Decimal, Symbolic)):
        return VecTuple([x])
    return x

def as_scalar(x) -> T | None:
    if isinstance(x, (int, float, Fraction, Decimal, Symbolic, bool)):
        return cast(T, x)
    elif isinstance(x, tuple) and len(x) == 1:
        return cast(T, x[0])
    return None

def as_scalar_strict(x) -> T:
    if isinstance(x, (int, float, Fraction, Decimal, Symbolic, str, bool)):
        return cast(T, x)
    elif isinstance(x, tuple) and len(x) == 1:
        return cast(T, x[0])
    raise NumericConversionError(f'The quantity {x} could not be converted to a numeric/symbolic scalar.')


#
# Numeric/Quantified VecTuples
#

class VecTuple(tuple[T, ...]):
    "A variant tuple type that supports addition and scalar multiplication like a vector."
    def __new__(cls, contents: Iterable[T]) -> 'VecTuple[T]':
        return super().__new__(cls, contents)     # type: ignore

    def __str__(self):
        return f'<{", ".join(map(str, self))}>'

    def __frplib_repr__(self):
        return self.__str__()

    def map(self, fn) -> 'VecTuple[T]':
        return self.__class__(map(fn, self))

    @property
    def dim(self):
        return len(self)

    @classmethod
    def show(cls, vtuple: 'VecTuple[T]', scalarize=True) -> str:
        if scalarize and len(vtuple) == 1:
            return str(vtuple[0])
        return str(vtuple)

    def __add__(self, other) -> 'VecTuple[T]':
        other = extend(other, len(self))  # Experimental: R style vector extension
        if not isinstance(other, tuple):
            return NotImplemented
        return VecTuple(map(add, self, other))

    def __radd__(self, other) -> 'VecTuple[T]':
        other = extend(other, len(self))  # Experimental: R style vector extension
        if not isinstance(other, tuple):
            return NotImplemented
        return VecTuple(map(add, other, self))

    def __sub__(self, other) -> 'VecTuple[T]':
        other = extend(other, len(self))  # Experimental: R style vector extension
        if not isinstance(other, tuple):
            return NotImplemented
        return VecTuple(map(sub, self, other))

    def __rsub__(self, other) -> 'VecTuple[T]':
        other = extend(other, len(self))  # Experimental: R style vector extension
        if not isinstance(other, tuple):
            return NotImplemented
        return VecTuple(map(sub, other, self))

    def __mul__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: x * z, self))     # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __rmul__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: z * x, self))     # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __truediv__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: x / z, self))  # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __floordiv__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: x // z, self))  # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __mod__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: x % z, self))  # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __pow__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: x ** z, self))  # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __rpow__(self, other) -> 'VecTuple[T]':
        z: T | None = as_scalar(other)   # type: ignore
        if z is not None:
            # Since other is Any in the superclass, the type system
            # cannot let us check that other has a matching type
            try:
                return VecTuple(map(lambda x: z ** x, self))  # type: ignore
            except Exception:
                return NotImplemented
        return NotImplemented

    def __matmul__(self, other) -> T:
        if not isinstance(other, Iterable):  # Allow vector like things
            return NotImplemented
        return reduce(add, map(mul, self, other), cast(T, 0))

    def __abs__(self):
        sq_norm = self @ self
        if isinstance(sq_norm, Symbolic):
            raise NotImplementedError('Symbolic function application not yet implemented')
        dot_prod = self @ self
        if isinstance(dot_prod, Symbolic):
            return symbolic_sqrt(dot_prod)
        elif isinstance(dot_prod, (int, Decimal)):
            return numeric_sqrt(dot_prod)
        return math.sqrt(dot_prod)

    def __getitem__(self, key):
        x = super().__getitem__(key)
        if isinstance(key, slice):
            return VecTuple(x)
        return x

    def __eq__(self, other):
        other = from_scalar(other)   # Allow scalar equality of VecTuples, no invariants changed
        try:
            return super().__eq__(other)
        except TypeError as e:
            raise OperationError(f'Could not test for == with {other}: {str(e)}')

    def __hash__(self):  # Need this because we play with __eq__
        return super().__hash__()  # Modified __eq__ does not change the invariant

    def __ne__(self, other):
        other = from_scalar(other)   # Allow scalar comparison of VecTuples
        try:
            return super().__ne__(other)
        except TypeError as e:
            raise OperationError(f'Could not test for != with {other}: {str(e)}')

    def __lt__(self, other):
        other = from_scalar(other)   # Allow scalar comparison of VecTuples
        try:
            return super().__lt__(other)
        except TypeError as e:
            raise OperationError(f'Could not test for != with {other}: {str(e)}')

    def __le__(self, other):
        other = from_scalar(other)   # Allow scalar comparison of VecTuples
        try:
            return super().__le__(other)
        except TypeError as e:
            raise OperationError(f'Could not test for != with {other}: {str(e)}')

    def __gt__(self, other):
        other = from_scalar(other)   # Allow scalar comparison of VecTuples
        try:
            return super().__gt__(other)
        except TypeError as e:
            raise OperationError(f'Could not test for != with {other}: {str(e)}')

    def __ge__(self, other):
        other = from_scalar(other)   # Allow scalar comparison of VecTuples
        try:
            return super().__ge__(other)
        except TypeError as e:
            raise OperationError(f'Could not test for != with {other}: {str(e)}')

    @classmethod
    def join(cls: Type[Self], values: Iterable[Self]) -> Self:
        combined = []
        for value in values:
            combined.extend(list(value))
        return cls(combined)


def vec_tuple(*a: T) -> VecTuple[T]:
    "Collects its arguments into a VecTuple"
    return VecTuple(a)

def as_vec_tuple(x: T | Iterable[T] = ()) -> VecTuple[T]:
    "Converts an iterable to (or wraps a single value in) a VecTuple"
    if isinstance(x, VecTuple):
        return x
    if isinstance(x, Iterable) and not isinstance(x, str):
        return VecTuple(x)
    return vec_tuple(x)

def as_numeric_vec(x):
    # ATTN: Consider using as_real here
    if isinstance(x, Iterable) and not isinstance(x, str):
        return VecTuple(map(scalar_as_numeric, x))
    else:
        return vec_tuple(scalar_as_numeric(x))

def is_vec_tuple(x) -> TypeGuard[VecTuple[T]]:
    "Is this a VecTuple?"
    return isinstance(x, VecTuple)
