# Numeric types that are interoperable but (ideally) readable and efficient
#
# These three categories can be used for probabilities or values.
# Prefer NumericF for probabilities, but this can get computationally costly,
# so considering using NumericD and post-converting to Fraction for output,
# conditionally on the denominator size. Get interpretable and accurate output
# without too much effort. Include these conversion functions here;
# See also VecTuple.py (formerly VecTupleSafe)

from __future__  import annotations

import math
import re

from abc               import abstractmethod
from collections.abc   import Iterable
from dataclasses       import dataclass
from decimal           import Decimal, ROUND_HALF_UP, ROUND_UP, ROUND_FLOOR, ROUND_CEILING
from enum              import Enum, auto
from fractions         import Fraction
from typing            import cast, Literal, Union
from typing_extensions import TypeAlias, TypeGuard

from frplib_pico.env        import environment
from frplib_pico.exceptions import EvaluationError


#
# Constants
#

DEFAULT_RATIONAL_DENOM_LIMIT = 1000000000   # Default Decimal -> Fraction conversion

class NumType(Enum):
    INTEGER = auto()
    RATIONAL = auto()
    REAL = auto()


#
# Numeric Quantities
#

class NumericQuantity:
    @abstractmethod
    def __float__(self) -> float:
        ...

    @abstractmethod
    def rational(self, limit_denominator=False) -> 'RationalQuantity':
        ...

    @abstractmethod
    def real(self) -> 'RealQuantity':
        ...

    def __lt__(self, other):
        if isinstance(other, NumericQuantity):
            return self.real() < other.real()
        return float(self) < other

    def __le__(self, other):
        if isinstance(other, NumericQuantity):
            return self.real() <= other.real()
        return float(self) <= other

    def __gt__(self, other):
        if isinstance(other, NumericQuantity):
            return self.real() > other.real()
        return float(self) > other

    def __ge__(self, other):
        if isinstance(other, NumericQuantity):
            return self.real() >= other.real()
        return float(self) >= other

@dataclass(frozen=True)
class IntegerQuantity(NumericQuantity):
    type: Literal[NumType.INTEGER] = NumType.INTEGER
    value: int = 0

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def rational(self, limit_denominator=False) -> 'RationalQuantity':
        return RationalQuantity(value=Fraction(self.value))

    def real(self) -> 'RealQuantity':
        return RealQuantity(value=Decimal(self.value))

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class RationalQuantity(NumericQuantity):
    type: Literal[NumType.RATIONAL] = NumType.RATIONAL
    value: Fraction = Fraction(0)

    def __float__(self) -> float:
        return float(self.value)

    def rational(self, limit_denominator=False) -> 'RationalQuantity':
        return self

    def real(self) -> 'RealQuantity':
        return RealQuantity(value=Decimal(self.value.numerator) / Decimal(self.value.denominator))

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class RealQuantity(NumericQuantity):
    type: Literal[NumType.REAL] = NumType.REAL
    value: Decimal = Decimal('0')

    def __float__(self) -> float:
        return float(self.value)

    def rational(self, limit_denominator=False) -> RationalQuantity:
        x = Fraction(self.value)
        if limit_denominator:
            x = x.limit_denominator(limit_denominator)
        return RationalQuantity(value=x)

    def real(self) -> 'RealQuantity':
        return self

    def __str__(self) -> str:
        return str(self.value)


#
# Numeric Types
#
# The NumericX types provide separate interoperable types for
# consistent arithmetic

NumericQ: TypeAlias = Union[IntegerQuantity, RationalQuantity, RealQuantity]
ScalarQ:  TypeAlias = Union[int, float, Fraction, Decimal, NumericQ, str]

NumericB: TypeAlias = Union[int, float]     # Binary floating point numbers
NumericD: TypeAlias = Union[int, Decimal]   # Decimal floating point numbers
NumericF: TypeAlias = Union[int, Fraction]  # Arbitrary Precision rational numbers

Numeric:  TypeAlias = NumericD  # Default underlying numeric representation

def is_scalar_q(x) -> TypeGuard[Union[int, float, Fraction, Decimal, NumericQ, str]]:
    return isinstance(x, (int, float, Fraction, Decimal, NumericQuantity, str, bool))  # bool auto cast to int

def is_numeric(x) -> TypeGuard[Union[int, Decimal]]:
    return isinstance(x, (int, float, Fraction, Decimal, NumericQuantity, str, bool))  # bool auto cast to int


#
# Numeric Conversion
#

REAL_ZERO = Decimal('0')
REAL_ONE = Decimal('1')
DECIMAL_DIG = 27  # current decimal precision used (digits)
NICE_DIGITS = 16  # used in nice_round; must be 0 <= _ <= DECIMAL_DIG with current Decimal setup

rat_denom = r'/(?:[1-9][0-9]{0,2}(?:_[0-9]{3})+|[1-9][0-9]*)'
decimal = r'\.[0-9]*'
sci_exp = r'[eE][-+]?(?:0|[1-9][0-9]*)'
opt_sign = r'-?'

integer_re = r'(?:0|[1-9][0-9]{0,2}(?:_[0-9]{3})+|[1-9][0-9]*)'  # _ separators allowed
numeric_re = rf'({opt_sign})({integer_re})(?:({rat_denom})|({decimal}(?:{sci_exp})?)|({sci_exp}))?'

def numeric_q_from_str(s: str) -> NumericQ:
    m = re.match(numeric_re, s.strip().replace('_', ''))
    if not m:
        m = re.match(r'(?i)-?inf(?:inity)?', s)
        if m:
            return RealQuantity(value=Decimal(s))  # +/- Infinity
        raise EvaluationError(f'Could not parse string as a numeric quantity: "{s}"')

    sign, integer, denom, dec_exp, exp = m.groups('')

    if not denom and not dec_exp and not exp:
        return IntegerQuantity(value=int(sign + integer))

    if denom:
        return RationalQuantity(value=Fraction(int(sign + integer), int(denom[1:])))

    if dec_exp:
        return RealQuantity(value=Decimal(sign + integer + dec_exp))

    return RealQuantity(value=Decimal(sign + integer + exp))

def numeric_q(
        x: ScalarQ = 0,
        exclude: Literal[NumType.RATIONAL] | Literal[NumType.REAL] | None = None,
        limit_denominator=DEFAULT_RATIONAL_DENOM_LIMIT
) -> NumericQ:
    if isinstance(x, str):
        x = numeric_q_from_str(x)    # Fall through

    if isinstance(x, IntegerQuantity):
        return x

    if isinstance(x, int):
        return IntegerQuantity(value=x)

    if isinstance(x, RationalQuantity):
        if exclude == NumType.RATIONAL:
            return x.real()
        return x

    if isinstance(x, RealQuantity):
        if exclude == NumType.REAL:
            return x.rational()
        return x

    if isinstance(x, Fraction):           # Check complexity to decide if real?
        qvalue = RationalQuantity(value=x)
        if exclude == NumType.RATIONAL:
            return qvalue.real()
        return qvalue

    if isinstance(x, float):
        val = nice_round(Decimal(x), NICE_DIGITS).normalize()
    else:
        val = x  # This is a Decimal

    rvalue = RealQuantity(value=val)
    if exclude == NumType.REAL:
        return rvalue.rational()
    return rvalue

def nice_round(d: Decimal, dig=NICE_DIGITS) -> Decimal:
    sign, digits, exp = d.as_tuple()
    dig = min(dig - 1, DECIMAL_DIG)  # ATTN: negative dig is ok but think about that case
    n = len(digits)
    if n <= dig + 1 or not isinstance(exp, int):  # exp can be n, N, or F
        return d
    true_exp = n - 1 + exp
    new_exp = true_exp - dig
    dtuple = (0, tuple([1] + ([0] * dig)), new_exp)
    return d.quantize(Decimal(dtuple), ROUND_HALF_UP)  # .normalize()

# Adjust these parameters for a default numerical quantification policy
def as_numeric(x: ScalarQ = RealQuantity()) -> Numeric:
    "A specialized version of `numeric` that defines a system-wide quantification policy."
    return cast(
        Union[IntegerQuantity, RealQuantity],
        numeric_q(x, exclude=NumType.RATIONAL, limit_denominator=1000000000)
    ).value

def as_rational(x: ScalarQ = RationalQuantity(), limit_denominator=False) -> Fraction:
    return numeric_q(x).rational(limit_denominator).value

def as_real(x: ScalarQ = RealQuantity()) -> Decimal:
    return numeric_q(x).real().value

def as_nice_numeric(x: ScalarQ = RealQuantity(), digits=NICE_DIGITS) -> Numeric:
    xn = as_numeric(x)
    if isinstance(xn, int):
        return xn
    return nice_round(xn, digits).normalize()


#
# Specialized Calculations
#

def numeric_sqrt(x: ScalarQ) -> Numeric:
    return as_real(x).sqrt()

def numeric_exp(x: ScalarQ) -> Numeric:
    return as_real(x).exp()

def numeric_ln(x: ScalarQ) -> Numeric:
    return as_real(x).ln()

def numeric_log10(x: ScalarQ) -> Numeric:
    return as_real(x).log10()

def numeric_log2(x: ScalarQ) -> Numeric:
    c = as_real(2).ln()
    return as_real(x).ln() / c

def numeric_abs(x: ScalarQ) -> Numeric:
    if isinstance(x, int):
        return abs(x)
    return as_real(x).copy_abs()

def numeric_floor(x: ScalarQ) -> Numeric:
    if isinstance(x, int):
        return x
    return as_real(x).quantize(REAL_ONE, ROUND_FLOOR)

def numeric_ceil(x: ScalarQ) -> Numeric:
    if isinstance(x, int):
        return x
    return as_real(x).quantize(REAL_ONE, ROUND_CEILING)


#
# Friendly Decimals
#
# This type behaves well at the repl, but to keep things light weight
# it only is useful to wrap the final values of a computation because
# Decimal operations return new Decimals.
#

class RealValue(Decimal):
    def __frplib_repr__(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        if environment.is_interactive:
            return str(self)
        return super().__repr__()

#
# Numeric output (needs improvement)
#

# ATTN: Move these to environment as settable options, the first two
DEC_DENOMINATOR_LIMIT: int = 10**9
MAX_DENOMINATOR_EXC = 50
EXCLUDE_DENOMINATOR = {10, 20, 25, 50, 100, 125, 250, 500, 1000}
ROUND_MASK = Decimal('1.000000000')

def nroundx(x: Numeric, mask=ROUND_MASK, rounding=ROUND_HALF_UP) -> Decimal:
    return Decimal(x).quantize(mask, rounding)

def nround(x: Numeric, mask=ROUND_MASK, rounding=ROUND_HALF_UP) -> Decimal:
    return nroundx(x, mask, rounding).normalize()

def as_frac(x: Numeric, denom_limit=DEC_DENOMINATOR_LIMIT) -> Fraction:
    return Fraction(x).limit_denominator(denom_limit)

def denom_rules(denominator, max_denom=MAX_DENOMINATOR_EXC, exclude=EXCLUDE_DENOMINATOR) -> bool:
    return denominator < max_denom and denominator not in exclude

def show_prob(p: Numeric) -> str:
    "Provide a human readable view of a fractional number in [0,1]."
    if isinstance(p, int):
        return str(p)

    pf = as_frac(p)
    if denom_rules(pf.denominator):
        return str(pf)

    return str(nround(p))

def show_numeric(
        x: Numeric,
        max_denom=MAX_DENOMINATOR_EXC,
        exclude_denoms=EXCLUDE_DENOMINATOR,
        rounding_mask=ROUND_MASK,
        rounding=ROUND_HALF_UP
) -> str:
    if isinstance(x, int):
        return str(x)

    frac = as_frac(x)
    if denom_rules(frac.denominator, max_denom, exclude_denoms):
        return str(frac)

    return str(nround(x, rounding_mask, rounding))

def show_nice_numeric(
        x: Numeric,
        max_denom=MAX_DENOMINATOR_EXC,
        exclude_denoms=EXCLUDE_DENOMINATOR,
        digits=NICE_DIGITS
) -> str:
    if isinstance(x, int):
        return str(x)

    frac = as_frac(x)
    if denom_rules(frac.denominator, max_denom, exclude_denoms):
        return str(frac)

    return str(nice_round(x, digits).normalize())

def show_values(
        xs: Iterable[Numeric],
        max_denom=MAX_DENOMINATOR_EXC,
        exclude_denoms=EXCLUDE_DENOMINATOR,
        rounding_mask=ROUND_MASK,
        rounding=ROUND_HALF_UP,
        digit_shift=5
) -> list[str]:
    "Find a pleasing common representation for a list of numeric quantities."
    # Do the values all have a simple common denominator?
    # If so, show as rationals with common denominator
    # ATTN: If there are *two* shared denominators that follow the rules, that would be useful
    xs = list(xs)
    ratl_xs = [as_rational(x, limit_denominator=DEC_DENOMINATOR_LIMIT) for x in xs]
    common_denom = math.lcm(*[r.denominator for r in ratl_xs])
    if common_denom == 1:
        return list(map(str, ratl_xs))
    elif denom_rules(common_denom, max_denom, exclude_denoms):
        return [f'{int(x.numerator * common_denom / x.denominator)}/{common_denom}' for x in ratl_xs]

    # Otherwise, show as real numbers, rounding all to a reasonable common scale
    real_xs = [as_real(x) for x in xs]
    expts = [x.copy_abs() for x in real_xs if x != REAL_ZERO]
    size = (sum(expts) / len(expts)).log10() if len(expts) > 0 else digit_shift  # type: ignore
    digits = digit_shift - int(nround(size, REAL_ONE, ROUND_UP))
    mask = Decimal('1.' + ('0' * digits)) if digits > 0 else REAL_ONE

    return [str(nice_round(x, 5)) for x in real_xs]
    # return [str(nround(x, mask)) for x in real_xs]

def show_tuples(
        tups: Iterable[tuple],
        max_denom=MAX_DENOMINATOR_EXC,
        exclude_denoms=EXCLUDE_DENOMINATOR,
        rounding_mask=ROUND_MASK,
        rounding=ROUND_HALF_UP,
        scalarize=True
) -> list[str]:
    "Convert a list of tuples to strings with angle-bracket syntax, with a shared representation for each component."
    # if dim == 1:
    #     return show_values([tup[0] for tup in tups], max_denom, exclude_denoms, rounding_mask, rounding)

    # Transpose, Format, and Transpose back
    outT = []
    for out in zip(*tups):  # , strict=True
        outT.append(show_values(out, max_denom, exclude_denoms, rounding_mask, rounding))
    dim = len(outT)
    if scalarize and dim == 1:
        return [components[0] for components in zip(*outT)]  # , strict=True
    return [f'<{", ".join(components)}>' for components in zip(*outT)]  # , strict=True

# Use VecTuple.show when value types are normalized here
def show_tuple(
        tup: tuple,
        max_denom=MAX_DENOMINATOR_EXC,
        exclude_denoms=EXCLUDE_DENOMINATOR,
        rounding_mask=ROUND_MASK,
        rounding=ROUND_HALF_UP,
        scalarize=True
) -> str:
    "Show a tuple with angle bracket syntax, but drop brackets for scalars."
    if scalarize and len(tup) == 1:
        return show_numeric(tup[0], max_denom, exclude_denoms, rounding_mask, rounding)
    components = [show_numeric(x, max_denom, exclude_denoms, rounding_mask, rounding)
                  for x in tup]
    return f'<{", ".join(components)}>'


# Older

skip_denoms = {5, 10, 50, 100, 500, 1000, 5000, 10000}
def show_num(number: int | float | Decimal | Fraction,
             real_only=False, denom_limit=1000000, dec_round='1.00000000') -> str:
    if isinstance(number, float):
        if not real_only:
            rational = Fraction.from_float(number).limit_denominator(denom_limit)
            if rational.denominator not in skip_denoms and abs(number - float(rational)) < 1.0e-7:
                return str(rational)
        return str(Decimal(number).quantize(Decimal(dec_round), rounding=ROUND_HALF_UP).normalize())
    return str(number)

# Use VecTuple.show when value types are normalized here
def old_show_tuple(tup: tuple, real_only=False, denom_limit=1000000, dec_round='1.000000') -> str:
    "Show a tuple with angle bracket syntax, but give values only for scalars."
    if len(tup) == 1:
        return show_num(tup[0], real_only, denom_limit, dec_round)
    return f'<{", ".join([show_num(x, real_only, denom_limit, dec_round) for x in tup])}>'
