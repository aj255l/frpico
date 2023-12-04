from __future__ import annotations

import re

from abc               import ABC
from collections       import defaultdict
from collections.abc   import Generator
from decimal           import Decimal
from functools         import reduce
from operator          import add
from typing            import cast, Literal, Union
from typing_extensions import TypeGuard

from frplib_pico.exceptions import ConstructionError
from frplib_pico.numeric    import (Numeric, ScalarQ,
                               as_real, as_nice_numeric, as_numeric,
                               is_scalar_q, show_numeric)
# from protocols  import NamedCallable
# from utils      import every

#
# Helpers
#

def merge_with(a, b, merge_fn=lambda x, y: y):
    merged = {k: a.get(k, b.get(k)) for k in a.keys() ^ b.keys()}
    merged.update({k: merge_fn(a[k], b[k]) for k in a.keys() & b.keys()})
    return merged

def is_zero(x: Numeric):  # Move to Numeric, should it take ScalarQ?
    return x == 0  # Temp

def show_coef(x: ScalarQ) -> str:
    return show_numeric(as_numeric(x), max_denom=1)

def purify(x: Symbolic) -> Symbolic | Numeric:
    pv = x.pure_value()
    if pv is not None:
        return pv
    return x

def common_term_rf(acc: dict[str, int] | None, multi: SymbolicMulti) -> dict[str, int]:
    """Reducing function that extracts common multi terms from a multi sum. Coefs excluded.

    Apply this to a SymbolicMultiSum x with reduce(common_term_rf, x). This returns
    a term object for a SymbolicMulti with the variables and their powers represented.
    An empty object corresponds to a constant, i.e., no nontrivial common term.

    """
    if acc is None:
        acc = multi.term.copy()
        return acc
    acc_prime = defaultdict(int)
    for var in acc:
        if var in multi.term:
            acc_prime[var] = min(acc[var], multi.term[var])
    return acc_prime

def common_term(sym_sum: SymbolicMultiSum) -> tuple[SymbolicMulti, SymbolicMultiSum]:
    "Extracts a common term (if any) from sum and return (term, adjusted sum)."
    term = reduce(common_term_rf, sym_sum.terms, None)
    assert term is not None

    if len(term) == 0:
        return (symbolic_one, sym_sum)

    common = SymbolicMulti.from_terms(term, sym_sum.coef)
    adj_terms = []
    for oterm in sym_sum.terms:
        adj, _, _ = multis_divide(oterm, common)
        adj_terms.append(adj / sym_sum.coef)
    adj_sum = SymbolicMultiSum(adj_terms)

    return (common, adj_sum)

def multis_divide(
        multi_num: SymbolicMulti,
        multi_den: SymbolicMulti
) -> tuple[SymbolicMulti, SymbolicMulti, bool]:
    """
    Returns None if the terms do not divide, or (1, denominator) or (numerator, 1)
    if they do, where the non-1 term is the non-canceled remaining term.

    """
    dpv = multi_den.pure_value()
    if dpv is not None:
        return (multi_num / dpv, symbolic_one, False)

    npv = multi_num.pure_value()
    if npv is not None:
        return (symbolic_one, multi_den / npv, False)

    if (multi_den.term.keys() <= multi_num.term.keys() or
       multi_num.term.keys() < multi_den.term.keys()):
        num = multi_num.term.copy()
        den = multi_den.term.copy()

        for k in multi_num.term.keys() & multi_den.term.keys():
            if den[k] <= num[k]:
                num[k] -= den[k]
                del den[k]
            else:
                den[k] -= num[k]
                del num[k]

        return (SymbolicMulti.from_terms(num, multi_num.coef),
                SymbolicMulti.from_terms(den, multi_den.coef),
                True)

    return (multi_num, multi_den, False)

def factor_quadratic(multisum: SymbolicMultiSum):
    pass


#
# Unique Symbols
#

def symbol_name_generator(base='#x_') -> Generator[str, None, None]:
    script = 0
    while True:
        yield f'{base}{{{script}}}'
        script += 1

gensym = symbol_name_generator()


#
# Generic Symbolic Quantities
#

class Symbolic(ABC):
    def is_pure(self):
        ...

    def pure_value(self):
        ...

    @property
    def key_of(self):
        ...

    @property
    def sort_key(self):
        ...

    def substitute(self, mapping: dict[str, ScalarQ], purify=True):
        ...

    def __lt__(self, other):
        "Sort symbols before numbers and by sort_key against Symbolics."
        if isinstance(other, Symbolic):
            return self.sort_key < other.sort_key
        if isinstance(other, str):
            return str(self) < other
        if is_scalar_q(other):
            pv = self.pure_value()
            if pv is not None:
                return pv < as_real(other)
        return True


#
# Multinomial Terms
#

class SymbolicMulti(Symbolic):
    "A symbolic multinomial term c a_1^k_1 a_2^k_2 ... a_n^k_n."
    def __init__(self, vars: list[str], powers: list[int], coef: ScalarQ = 1):
        # if powers too short, those count as 0, so ok; extra powers ignored
        # coef, [], []  acts as a scalar and a *multiplicative* identity
        self.coef = as_numeric(coef)

        order = 0
        sig = '1'  # show_coef(self.coef)
        multi: dict[str, int] = defaultdict(int)
        if not is_zero(self.coef):
            for var, pow in zip(vars, powers):
                if not var:
                    raise ConstructionError('A symbolic variable name must be a non-empty string.')
                multi[var] += pow
                order += pow
            sigs = []
            for var, pow in sorted(multi.items()):
                if pow == 0:
                    del multi[var]
                    continue
                sigs.append(f'{var}^{pow}')
            if sigs:
                sig = " ".join(sigs)

        self.term = multi
        self.order = order
        self.key = sig
        self.as_str: Union[str, None] = None   # Computed lazily

    @classmethod
    def from_terms(cls, multi: dict[str, int], coef: Numeric = 1):
        if is_zero(coef):
            return SymbolicMulti([], [], 0)
        vars = []
        pows = []
        for var, pow in multi.items():
            if pow != 0:
                vars.append(var)
                pows.append(pow)
        return cls(vars, pows, coef)

    @classmethod
    def pure(cls, x: ScalarQ = 0):
        return cls([], [], as_numeric(x))

    def is_pure(self) -> bool:
        return len(self.term) == 0

    def pure_value(self):
        if self.is_pure():
            return self.coef
        return None

    def substitute(self, mapping: dict[str, ScalarQ], purify=True) -> SymbolicMulti | Numeric:
        coef = self.coef
        multi = self.term.copy()
        for var, pow in self.term.items():
            if var in mapping:
                coef *= as_numeric(mapping[var]) ** pow
                del multi[var]
        coef = as_numeric(coef)  # ATTN: needed?
        if len(multi) == 0:
            return coef
        return SymbolicMulti.from_terms(multi, coef)

    def reset_coef(self, new_coef):
        return SymbolicMulti.from_terms(self.term, new_coef)

    @property
    def signature(self) -> str:
        return self.key

    @property
    def key_of(self):
        return self.key

    @property
    def sort_key(self):
        return (self.coef, self.key)

    def __bool__(self) -> bool:
        return False if self.is_pure() and is_zero(self.coef) else True

    def __str__(self) -> str:
        if self.as_str is None:
            if self.is_pure():
                self.as_str = show_coef(self.coef)
            else:
                coef = '' if self.coef == 1 else show_coef(self.coef) + ' '
                term = re.sub(r'\^1( |$)', lambda m: ' ' if m.group(1) else '', self.signature)
                self.as_str = coef + term
        return self.as_str

    def __frplib_repr__(self) -> str:
        return str(self)

    def clone(self) -> SymbolicMulti:
        return self.from_terms(self.term, self.coef)

    def __mul__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if is_zero(as_numeric(other)):
                return 0
            return self.from_terms(self.term, self.coef * as_real(other))
        if isinstance(other, SymbolicMulti):
            term = other.term if self.is_pure() else merge_with(self.term, other.term, add)
            coef = self.coef * other.coef
            return SymbolicMulti.from_terms(term, coef)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if is_zero(as_numeric(other)):
                return 0
            return self.from_terms(self.term, as_real(other) * self.coef)
        # Cannot be SymbolicMulti in rul
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if self.is_pure():
                return self.from_terms(self.term, self.coef + as_real(other))
            return SymbolicMultiSum([self, SymbolicMulti.pure(as_numeric(other))])
        if isinstance(other, SymbolicMulti):
            return SymbolicMultiSum([self, other])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if self.is_pure():
                return self.from_terms(self.term, as_real(other) + self.coef)
            return SymbolicMultiSum([SymbolicMulti.pure(as_numeric(other)), self])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if self.is_pure():
                return self.from_terms(self.term, self.coef - as_real(other))
            return SymbolicMultiSum([self, SymbolicMulti.pure(as_numeric(-other))])
        if isinstance(other, SymbolicMulti):
            if self.key_of == other.key_of:
                return as_numeric(self.coef - other.coef)
            return SymbolicMultiSum([self, -1 * other])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if self.is_pure():
                return self.from_terms(self.term, as_real(other) - self.coef)
            return SymbolicMultiSum([SymbolicMulti.pure(as_numeric(other)), -1 * self])
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            return self.from_terms(self.term, self.coef / as_real(other))

        if isinstance(other, (SymbolicMulti, SymbolicMultiSum)):
            return symbolic(self, other)

        if isinstance(other, SymbolicMultiRatio):
            return symbolic(self * other.denominator, other.numerator)

        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            return symbolic(SymbolicMulti.pure(other), self)

        if isinstance(other, (SymbolicMulti, SymbolicMultiSum)):
            return symbolic(other, self)

        if isinstance(other, SymbolicMultiRatio):
            return symbolic(other.numerator, self * other.denominator)

        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, int):
            if n == 0:
                return 1
            term = self.term.copy()
            for var, pow in term.items():
                term[var] = pow * n if var else pow   # Constant always has power 1
            return self.from_terms(term, self.coef ** n)
        return NotImplemented

symbolic_zero = SymbolicMulti.pure(0)
symbolic_one = SymbolicMulti.pure(1)


#
# Symbolic Function Calls
#

# class SymbolicFn(SymbolicMulti):
#     def __init__(self, fn: NamedCallable, args: list[Symbolic]) -> None:
#         var = next(gensym)
#         self.fns = {var: {'fn': fn, 'args': args[:]}}
#         super().__init__([var], [1])
#
#     def is_pure(self) -> bool:
#         return (super().is_pure() or
#                 (self.term.keys() == self.fns.keys() and
#                  all(every(lambda s: s.is_pure(), f['args']) for f in self.fns.values())))
#
#     def pure_value(self):
#         if self.is_pure():
#             value = self.coef
#             for fn_key in self.fns:
#                 fstar = self.fns[fn_key]
#                 fn = cast(Callable[..., Numeric], fstar['fn'])
#                 args: list = fstar['args']
#                 assert callable(fn)
#                 value *= as_real(fn(*args))
#             return value
#         return None
#
#     def substitute(self, mapping: dict[str, ScalarQ], purify=True) -> SymbolicFn | Numeric:
#         multi = self.substitute(mapping, purify)
#         pass
#
#     def __mul__(self, other):
#         # Stay as SymbolicFn against SymbolicMulti and pure
#         pass
#
#     def __rmul__(self, other):
#         # Stay as SymbolicFn against SymbolicMulti and pure
#         pass


#
# Sums of Multinomial Terms
#

class SymbolicMultiSum(Symbolic):
    """A sum of symbolic multinomial terms sum_i c_i a_i1^k_1 a_i2^k_2 ... a_in^k_n.

    """
    def __init__(self, multis: list[SymbolicMulti]) -> None:
        terms = self.combine_terms(multis)
        if not terms:
            self.terms: list[SymbolicMulti] = []
            self.coef: Numeric = 0
            self.order = 0
            self.key = '0'
            self.as_str: str | None = '0'
        elif len(terms) == 1 and terms[0].is_pure():
            self.terms = [terms[0]]
            self.coef = terms[0].pure_value()
            self.order = 0
            self.key = '1'
            self.as_str = show_coef(self.coef)
        else:
            terms = sorted(terms, key=lambda t: t.signature)
            self.order = 0
            for term in terms:
                if term.order > self.order:
                    self.order = term.order
                    self.coef = term.coef
            self.key = " + ".join([str(term / self.coef) for term in terms])
            self.terms = terms
            self.as_str = None

    def __str__(self) -> str:
        if self.as_str is None:
            self.as_str = " + ".join([str(term) for term in self.terms])
        return self.as_str

    def __frplib_repr__(self) -> str:
        return str(self)

    def is_pure(self) -> bool:
        return len(self.terms) == 1 and self.terms[0].is_pure()

    def is_single(self) -> bool:
        return len(self.terms) == 1

    def pure_value(self):
        if self.is_pure():
            return self.coef
        return None

    def substitute(self, mapping: dict[str, ScalarQ], purify=True) -> SymbolicMultiSum | Numeric:
        if self.is_pure() and purify:
            return as_nice_numeric(self.coef)
        total: Union[SymbolicMultiSum, Numeric] = sum(term.substitute(mapping, purify=purify)  # type: ignore
                                                      for term in self.terms)
        if isinstance(total, Symbolic):
            return total
        return as_nice_numeric(total)

    @property
    def key_of(self):
        return self.key

    @property
    def sort_key(self):
        return (self.coef, self.key)

    @classmethod
    def singleton(cls, sym: SymbolicMulti):
        return SymbolicMultiSum([sym])

    @staticmethod
    def combine_terms(terms: list[SymbolicMulti]) -> list[SymbolicMulti]:
        combined: dict[str, SymbolicMulti] = {}
        for term in terms:
            k = term.signature
            if k in combined:
                combined[k] = term.reset_coef(combined[k].coef + term.coef)
            else:
                combined[k] = term
        return [v for v in combined.values() if not is_zero(v.coef)]

    def __add__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            return SymbolicMultiSum([*self.terms, SymbolicMulti.pure(other)])

        if isinstance(other, SymbolicMulti):
            return SymbolicMultiSum([*self.terms, other])

        if isinstance(other, SymbolicMultiSum):
            return SymbolicMultiSum([*self.terms, *other.terms])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            return SymbolicMultiSum([*self.terms, SymbolicMulti.pure(-other)])

        if isinstance(other, SymbolicMulti):
            return SymbolicMultiSum([*self.terms, -1 * other])

        if isinstance(other, SymbolicMultiSum):
            terms = [*self.terms]
            terms.extend(-1 * term for term in other.terms)
            return SymbolicMultiSum(terms)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            return SymbolicMultiSum([SymbolicMulti.pure(other), *self.terms])

        if isinstance(other, SymbolicMulti):
            return SymbolicMultiSum([other, *self.terms])

        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            terms = [SymbolicMulti.pure(other)]
            terms.extend(-1 * term for term in self.terms)
            return SymbolicMultiSum(terms)

        if isinstance(other, SymbolicMulti):
            terms = [other]
            terms.extend(-1 * term for term in self.terms)
            return SymbolicMultiSum(terms)

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            nother = as_numeric(other)
            if is_zero(nother):
                return 0
            if nother == 1:
                return self
            pv = self.pure_value()
            if pv is not None:
                return pv * nother
            return SymbolicMultiSum([x * as_real(nother) for x in self.terms])

        pv = self.pure_value()
        if pv is not None:
            return pv * other

        if isinstance(other, SymbolicMulti):
            return SymbolicMultiSum([x * other for x in self.terms])

        if isinstance(other, SymbolicMultiSum):
            ov = other.pure_value()
            if ov is not None:
                return self * ov

            # Combine like terms
            combined: dict[str, SymbolicMulti] = {}
            for term1 in self.terms:
                for term2 in other.terms:
                    prod = term1 * term2
                    k = prod.signature
                    if k in combined:
                        combined[k].coef += prod.coef
                    else:
                        combined[k] = prod
            terms = [v for v in combined.values() if not is_zero(v.coef)]
            return SymbolicMultiSum(terms)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            nother = as_numeric(other)
            if is_zero(nother):
                return SymbolicMultiSum.singleton(symbolic_zero)
            if nother == 1:
                return self
            pv = self.pure_value()
            if pv is not None:
                return SymbolicMultiSum.singleton(SymbolicMulti.pure(nother * pv))
            return SymbolicMultiSum([as_real(nother) * x for x in self.terms])

        pv = self.pure_value()
        if pv is not None:
            return other * pv

        if isinstance(other, SymbolicMulti):
            return SymbolicMultiSum([other * x for x in self.terms])

        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, int):
            if n == 0:
                return SymbolicMultiSum.singleton(symbolic_one)
            if n == 1:
                return self
            if n % 2 == 0:
                return (self * self) ** (n // 2)
            return self * (self * self) ** (n // 2)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if other == 1:
                return self
            d = as_real(other)
            return SymbolicMultiSum([term / d for term in self.terms])

        if isinstance(other, (SymbolicMulti, SymbolicMultiSum, SymbolicMultiRatio)):
            opv = other.pure_value()
            if opv is not None:
                if opv == 1:
                    return self
                return SymbolicMultiSum([term / opv for term in self.terms])
            return simplify(symbolic(self, other))

        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if as_numeric(other) == 1:
                return symbolic(symbolic_one, self)
            d = as_real(other)
            return symbolic(SymbolicMulti.pure(d), self)

        if isinstance(other, (SymbolicMulti, SymbolicMultiRatio)):
            return simplify(symbolic(other, self))

        return NotImplemented


#
# Ratios of Sums of Multinomial Terms
#

class SymbolicMultiRatio(Symbolic):
    def __init__(self, numerator: SymbolicMultiSum, denominator: SymbolicMultiSum):
        terms = [numerator, denominator]
        # Simplify in the generic symbolic constructor so we can return
        # other types of values.
        self.terms = terms
        self.key = f'({numerator.key_of})/({denominator.key_of})'

        if numerator.is_pure() or numerator.is_single():
            num = str(numerator)
        else:
            num = f'({str(numerator)})'

        if denominator.is_pure() or denominator.is_single():
            den = str(denominator)
            if ' ' in den:
                den = f'({str(denominator)})'
        else:
            den = f'({str(denominator)})'

        self.as_str = f'{num}/{den}'

    @property
    def numerator(self):
        return self.terms[0]

    @property
    def denominator(self):
        return self.terms[1]

    @property
    def key_of(self):
        return self.key

    @property
    def sort_key(self):
        return (as_real(self.numerator.coef) / as_real(self.denominator.coef), self.key)

    def __str__(self) -> str:
        return self.as_str

    def __frplib_repr__(self) -> str:
        return str(self)

    def is_pure(self) -> bool:
        return self.numerator.is_pure() and self.denominator.is_pure()

    def pure_value(self):
        npv = self.numerator.pure_value()
        dpv = self.denominator.pure_value()

        if npv is not None and dpv is not None:
            return as_numeric(npv / dpv)
        return None

    def substitute(self, mapping: dict[str, ScalarQ], purify=True) -> Symbolic | Numeric:
        spv = self.pure_value()
        if spv is not None:
            return spv if purify else SymbolicMultiSum.singleton(SymbolicMulti.pure(spv))

        num = self.numerator.substitute(mapping, purify)
        den = self.denominator.substitute(mapping, purify)

        if not isinstance(num, SymbolicMultiSum) and not isinstance(num, SymbolicMultiSum):
            return as_nice_numeric(as_real(num / den))
        return simplify(symbolic(num, den))

    def __add__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if is_zero(as_numeric(other)):
                simplify(self)
            numer = self.numerator + as_real(other) * self.denominator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiRatio):
            if self.denominator.key_of == other.denominator.key_of:
                r = self.denominator.coef / other.denominator.coef
                return simplify(SymbolicMultiRatio(self.numerator + r * other.numerator, self.denominator))

            numer = self.numerator * other.denominator + self.denominator * other.numerator
            denom = self.denominator * other.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiSum):
            numer = self.numerator + other * self.denominator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMulti):
            if not other:
                return self
            numer = self.numerator + SymbolicMultiSum.singleton(other) * self.denominator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if is_zero(as_numeric(other)):
                self
            numer = as_real(other) * self.denominator + self.numerator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiSum):
            numer = other * self.denominator + self.numerator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMulti):
            if not other:
                return self
            numer = SymbolicMultiSum.singleton(other) * self.denominator + self.numerator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        return NotImplemented

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if is_zero(as_numeric(other)):
                return symbolic_zero
            numer = self.numerator * as_real(other)
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiRatio):
            if self.numerator.key_of == other.denominator.key_of:
                return simplify(symbolic(other.numerator, self.denominator))

            if self.denominator.key_of == other.numerator.key_of:
                return simplify(symbolic(self.numerator, other.denominator))

            numer = self.numerator * other.numerator
            denom = self.denominator * other.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiSum):
            numer = self.numerator * other
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMulti):
            numer = self.numerator * SymbolicMultiSum.singleton(other)
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if is_zero(as_numeric(other)):
                return symbolic_zero
            numer = as_real(other) * self.numerator
            denom = self.denominator
            return simplify(symbolic(numer, denom))

        if isinstance(other, SymbolicMultiSum):
            numer = other * self.numerator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMulti):
            numer = SymbolicMultiSum.singleton(other) * self.numerator
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            if other == 1:
                return self
            numer = self.numerator / as_numeric(other)
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiRatio):
            numer = self.numerator * other.denominator
            denom = self.denominator * other.numerator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMultiSum):
            numer = self.numerator
            denom = self.denominator * other
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMulti):
            numer = self.numerator
            denom = self.denominator * SymbolicMultiSum.singleton(other)
            return simplify(SymbolicMultiRatio(numer, denom))

        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Decimal)):   # is_scalar_q(other):
            numer = SymbolicMultiSum.singleton(SymbolicMulti.pure(as_numeric(other)))
            denom = self.denominator
            return simplify(SymbolicMultiRatio(numer * denom, self.numerator))

        if isinstance(other, SymbolicMultiSum):
            numer = self.denominator * other
            denom = self.numerator
            return simplify(SymbolicMultiRatio(numer, denom))

        if isinstance(other, SymbolicMulti):
            numer = self.denominator * SymbolicMultiSum.singleton(other)
            denom = self.numerator
            return simplify(SymbolicMultiRatio(numer, denom))

        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, int):
            if n == 0:
                return 1
            if n == 1:
                return self
            return simplify(symbolic(self.numerator ** n, self.denominator ** n))
        return NotImplemented


#
# Simple Simplification Rules
#

def simplify(a: Symbolic | Numeric) -> Union[Symbolic, Numeric]:
    if isinstance(a, (int, Decimal)):
        return a

    apv = a.pure_value()
    if apv is not None:
        return as_numeric(apv)

    if isinstance(a, SymbolicMulti):
        return a

    if isinstance(a, SymbolicMultiSum):
        return a

    assert isinstance(a, SymbolicMultiRatio)
    npv = a.numerator.pure_value()
    dpv = a.denominator.pure_value()

    if npv is not None:
        if is_zero(npv):
            return 0
        return SymbolicMultiRatio(SymbolicMultiSum.singleton(symbolic_one), a.denominator / as_real(npv))
    if dpv is not None:
        return a.numerator / as_real(dpv)

    # We have a rational with neither term pure
    # First try to extract and cancel common terms
    common_num, adj_num = common_term(a.numerator)
    common_den, adj_den = common_term(a.denominator)
    simple_num, simple_den, changed = multis_divide(common_num, common_den)

    if adj_num.key_of == adj_den.key_of:
        ratio = as_real(adj_num.coef / adj_den.coef)
        return purify(simple_num) * ratio / purify(simple_den)   # type: ignore

    if not changed:
        return a

    return (purify(simple_num) / purify(simple_den)) * (adj_num / adj_den)   # type: ignore

    # Other simplification rules?

    # if a.numerator.key_of == a.denominator.key_of:
    #     return as_real(a.numerator.coef / a.denominator.coef)

    # return a


#
# Symbolic Constructors (use these not the class constructors)
#

def symbolic(numerator: Union[Symbolic, str], denominator: Union[Symbolic, Literal[1]] = 1) -> Union[Symbolic, Numeric]:
    if isinstance(numerator, str):
        numerator = symbol(numerator)

    npv = numerator.pure_value()
    dpv = denominator.pure_value() if denominator != 1 else 1
    if npv is not None and dpv is not None:
        return as_real(npv / dpv)
    elif npv is not None and denominator == 1:
        return npv
    elif npv is not None:
        numerator = symbolic_one
        denominator = denominator / npv
    elif denominator == 1:
        return simplify(numerator)

    assert isinstance(numerator, Symbolic)
    assert isinstance(denominator, Symbolic)

    if isinstance(numerator, SymbolicMulti):
        numerator = SymbolicMultiSum.singleton(numerator)

    if isinstance(denominator, SymbolicMulti):
        denominator = SymbolicMultiSum.singleton(denominator)

    num_rat = isinstance(numerator, SymbolicMultiRatio)
    den_rat = isinstance(denominator, SymbolicMultiRatio)

    if num_rat and den_rat and numerator.key_of == denominator.key_of:        # type: ignore
        return ((numerator.numerator.coef * denominator.denominator.coef) /   # type: ignore
                (numerator.denominator.coef * denominator.numerator.coef))    # type: ignore
    if num_rat:
        numerator = cast(SymbolicMultiSum, numerator)
        return simplify(numerator.__truediv__(denominator))
    elif den_rat:
        denominator = cast(SymbolicMultiSum, denominator)
        return simplify(denominator.__rtruediv__(numerator))

    numerator = cast(SymbolicMultiSum, numerator)
    denominator = cast(SymbolicMultiSum, denominator)

    if numerator.key_of == denominator.key_of:
        return as_real(numerator.coef / denominator.coef)

    return simplify(SymbolicMultiRatio(numerator, denominator))


#
# Symbol constructors
#

def gen_symbol() -> SymbolicMulti:
    "Generates a symbol with a unique name."
    var = next(gensym)
    return SymbolicMulti([var], [1])

def symbol(var: str) -> SymbolicMulti:
    "Generates a symbol with the given name, typically a single letter."
    return SymbolicMulti([var], [1])

def is_symbolic(obj) -> TypeGuard[SymbolicMulti | SymbolicMultiSum | SymbolicMultiRatio]:
    return isinstance(obj, (SymbolicMulti, SymbolicMultiSum, SymbolicMultiRatio))


#
# Calculation Routines (may be subsumed by SymbolicFn)
#

def symbolic_sqrt(x: Symbolic) -> Symbolic:
    raise ConstructionError(f'Symbolic square root not yet implemented: {str(x)}')


#
# Info tags
#

setattr(symbol, '__info__', 'utilities::symbols')
setattr(gen_symbol, '__info__', 'utilities::symbols')
