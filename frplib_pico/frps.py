from __future__ import annotations

import math
import random

from abc               import ABC, abstractmethod
from collections       import defaultdict
from collections.abc   import Iterable
from typing            import Callable, cast, overload, Union
from typing_extensions import Self, Any, TypeAlias

from rich.table        import Table
from rich              import box as rich_box
from rich.panel        import Panel

from frplib_pico.env        import environment
from frplib_pico.exceptions import (ConditionMustBeCallable, ComplexExpectationWarning,
                               ConstructionError, FrpError, KindError, MismatchedDomain,)
from frplib_pico.kinds      import Kind, kind, ConditionalKind, permutations_of
# ATTN:Will replace this value type with a unified type TBD
from frplib_pico.numeric    import Numeric, show_tuple, as_real
from frplib_pico.protocols  import Projection, SupportsExpectation
from frplib_pico.quantity   import as_quant_vec
from frplib_pico.statistics import Statistic, compose2, infinity, tuple_safe
from frplib_pico.symbolic   import Symbolic
from frplib_pico.utils      import scalarize
from frplib_pico.vec_tuples import VecTuple, as_scalar, as_vec_tuple, vec_tuple


# ValueType: TypeAlias = VecTuple[Numeric]  # ATTN
QuantityType: TypeAlias = Union[Numeric, Symbolic]
ValueType: TypeAlias = VecTuple[QuantityType]  # ATTN


#
# Helpers
#


#
# FRP.sample can return either all the sampled values
# or a summary table. The latter is the default and
# is basically just a dict[ValueType, int], but we
# enhance this with representations and some other
# information
#

FrpDemo: TypeAlias = list[ValueType]

class FrpDemoSummary:
    # At most one of these should be non-None
    def __init__(
            self,
            *,
            summary: Self | dict[ValueType, int] | None = None,
            sample_size: int | None = None,
            samples: FrpDemo | None = None
    ) -> None:
        self._summary: dict[ValueType, int] = defaultdict(int)
        self._size = 0

        if summary and isinstance(summary, FrpDemoSummary):
            self._size = summary._size
            self._summary = {k: v for k, v in summary._summary.items()}
        elif summary and isinstance(summary, dict):
            self._size = len(summary)
            self._summary = {k: v for k, v in summary.items()}
        elif samples:
            for sample in samples:
                self.add(sample)

    def add(self, value: ValueType) -> Self:
        self._summary[value] += 1
        self._size += 1
        return self

    def rich_table(self, title: str | None = None):
        # ATTN: Put styles in a more central place (environment?), e.g., environment.styles['values']
        if title is None:
            title = 'Summary of Output Values'
        table = Table(title=title, box=rich_box.SQUARE_DOUBLE_HEAD)
        table.add_column('Values', justify='left', style='#4682b4', no_wrap=True)
        table.add_column('Count', justify='right')
        table.add_column('Proportion', justify='right', style='#6a6c6e')

        values = sorted(self._summary.keys())
        n = float(self._size)
        for value in values:
            table.add_row(show_tuple(value.map(lambda x: "{0:.4g}".format(x))),
                          str(self._summary[value]),
                          "{0:.4g}%".format(round(100 * self._summary[value] / n, 6)))

        return table

    def ascii_table(self, title: str | None = None) -> str:
        out: list[str] = []
        if title is None:
            title = 'Summary of output values:'
        out.append(title)

        values = sorted(self._summary.keys())
        n = float(self._size)
        widths = {'value': 0, 'count': 0, 'prop': 0}
        rows = []
        for value in values:
            cells = {
                'value': show_tuple(value.map(lambda x: "{0:.5g}".format(x))),  # str(VecTuple(value)),
                'count': "{0:,d}".format(self._summary[value]),
                'prop': "({0:.4f}%)".format(round(100 * self._summary[value] / n, 6))
            }
            rows.append(cells)
            widths = {k: max(len(cells[k]), widths[k]) for k in widths}
        for row in rows:
            out.append("{value:<{w[0]}s}    {count:>{w[1]}s}"
                       "    {prop:>{w[2]}s}".format(**row, w=list(widths.values())))
        return "\n".join(out)

    def table(self, ascii=False, title: str | None = None) -> str:
        if ascii:
            return self.ascii_table(title)
        return self.rich_table(title)

    def __len__(self) -> int:
        return self._size

    def __frplib_repr__(self):
        return self.table(environment.ascii_only)

    def __str__(self) -> str:
        return self.table(ascii=True)

    def __repr__(self) -> str:
        if environment.is_interactive:
            return str(self)
        return f'{self.__class__.__name__}(summary={repr(self._summary)}, sample_size={self._size})'


#
# FRP Expressions
#
# When an FRP is constructed from an expression rather than a kind,
# we record the expression (including the FRP objects themselves,
# which are logically immutable and thus held) and use that as a recipe
# for both sampling and constructing a value.
#

class FrpExpression(ABC):
    def __init__(self) -> None:
        self._cached_kind: Kind | None = None
        self._cached_value: ValueType | None = None

    @abstractmethod
    def sample1(self) -> ValueType:
        "Draw a sample from the underlying FRP's kind or get its value"
        ...

    @abstractmethod
    def value(self) -> ValueType:
        "Draw a sample from the underlying FRP's kind or get its value"
        ...

    @abstractmethod
    def kind(self) -> Kind:
        ...

    @abstractmethod
    def clone(self) -> FrpExpression:
        ...

class TransformExpression(FrpExpression):
    def __init__(
            self,
            transform: Callable[[ValueType], ValueType],
            target: FrpExpression
    ) -> None:
        super().__init__()
        self._transform = transform   # This will typically be a statistic
        self._target = target

    def sample1(self) -> ValueType:
        return self._transform(self._target.sample1())

    def value(self) -> ValueType:
        if self._cached_value is None:
            try:
                self._cached_value = self._transform(self._target.value())
            except Exception:  # ATTN: might be easiest to just evaluate the value at transform time
                if isinstance(self._transform, Statistic):
                    label = self._transform.name   # type: ignore
                else:
                    label = str(self._transform)
                raise MismatchedDomain(f'Statistic {label} is incompatible with this FRP,'
                                       f'could not evaluate it on the FRPs value.')
        return self._cached_value

    def kind(self) -> Kind:
        if self._cached_kind is None:
            self._cached_kind = self._target.kind() ^ self._transform
        return self._cached_kind

    def clone(self) -> 'TransformExpression':
        new_expr = TransformExpression(self._transform, self._target.clone())
        new_expr._cached_kind = self._cached_kind
        return new_expr

class IMixtureExpression(FrpExpression):
    def __init__(self, terms: Iterable['FrpExpression']) -> None:
        super().__init__()
        self._operands = list(terms)

        # Cache kind or value if appropriate
        # We only cache these if they are available for every term.
        # Moreover, we ensure that the kind is not too large,
        # as determined by FRP's complexity threshold.
        # We stop as soon as these conditions are not satisfied.
        threshold = math.log2(FRP.COMPLEXITY_THRESHOLD)
        logsize = 0.0
        cache_kind = True
        cache_value = True

        combined_values: list = []
        combined_kind = Kind.empty
        for f in self._operands:
            if cache_value:
                if f._cached_value is not None:
                    combined_values.extend(f._cached_value)
                else:
                    cache_value = False
            if cache_kind:
                if f._cached_kind is not None:
                    logsize += math.log2(f._cached_kind.size)
                    if logsize <= threshold:
                        combined_kind = combined_kind * f._cached_kind
                    else:
                        cache_kind = False
                else:
                    cache_kind = False
            elif not cache_value:
                break

        if cache_value:
            self._cached_value = as_quant_vec(combined_values)
        if cache_kind:
            self._cached_kind = combined_kind

    def sample1(self) -> ValueType:
        if len(self._operands) == 0:
            return VecTuple(())
        return join_values(operand.sample1() for operand in self._operands)

    def value(self) -> ValueType:
        if len(self._operands) == 0:
            return VecTuple(())
        if self._cached_value is None:
            self._cached_value = join_values(operand.value() for operand in self._operands)
        return self._cached_value

    def kind(self) -> Kind:
        if len(self._operands) == 0:
            return Kind.empty
        if self._cached_kind is None:
            kinds = [operand.kind() for operand in self._operands]
            combined_kind = Kind.empty
            for child in kinds:
                combined_kind = combined_kind * child
            self._cached_kind = combined_kind
        return self._cached_kind

    def clone(self) -> 'IMixtureExpression':
        new_expr = IMixtureExpression([term.clone() for term in self._operands])
        new_expr._cached_kind = self._cached_kind
        return new_expr

    def expectation(self):
        cached = [k._cached_kind for k in self._operands]
        if all(k is not None for k in cached):
            return as_vec_tuple([k.expectation() for k in cached])   # type: ignore
        elif all(isinstance(term, SupportsExpectation) for term in self._operands):
            return as_vec_tuple([term.expectation() for term in self._operands])  # type: ignore
        raise ComplexExpectationWarning('The expectation of this FRP could not be computed '
                                        'without first finding its kind.')

    @classmethod
    def append(cls, mixture: 'IMixtureExpression', other: 'FrpExpression') -> 'IMixtureExpression':
        "Returns a new IMixture with target as the last term."
        return IMixtureExpression([*mixture._operands, other])

    @classmethod
    def prepend(cls, mixture: 'IMixtureExpression', other: 'FrpExpression') -> 'IMixtureExpression':
        "Returns a new IMixture with target as the last term."
        return IMixtureExpression([other, *mixture._operands])

    @classmethod
    def join(cls, mixture1: 'IMixtureExpression', mixture2: 'IMixtureExpression') -> 'IMixtureExpression':
        "Returns a new IMixture with target as the last term."
        return IMixtureExpression([*mixture1._operands, *mixture2._operands])

class IMixPowerExpression(FrpExpression):
    def __init__(self, term: 'FrpExpression', pow: int) -> None:
        super().__init__()
        self._term = term
        self._pow = pow
        if (term._cached_kind is not None and
           pow * math.log2(term._cached_kind.size) <= math.log2(FRP.COMPLEXITY_THRESHOLD)):
            self._cached_kind = term._cached_kind ** pow

    def sample1(self) -> ValueType:
        draws = [self._term.sample1() for _ in range(self._pow)]
        return join_values(draws)

    def value(self) -> ValueType:
        if self._cached_value is None:
            self._cached_value = self.sample1()
        return self._cached_value

    def kind(self) -> Kind:
        if self._cached_kind is None:
            self._cached_kind = self._term.kind() ** self._pow
        return self._cached_kind

    def clone(self) -> 'IMixPowerExpression':
        new_expr = IMixPowerExpression(self._term.clone(), self._pow)
        new_expr._cached_kind = self._cached_kind
        return new_expr

    def expectation(self):
        if self._term._cached_kind is not None:
            exp = self._term._cached_kind.expectation()
            return as_vec_tuple([exp] * self._pow)
        elif isinstance(self._term, SupportsExpectation):
            exp = self._term.expectation()
            return as_vec_tuple([exp] * self._pow)
        raise ComplexExpectationWarning('The expectation of this FRP could not be computed '
                                        'without first finding its kind.')

class MixtureExpression(FrpExpression):
    # ATTN: the target should be passed to conditional_frp before this
    def __init__(self, mixer: FrpExpression, target: 'ConditionalFRP') -> None:
        super().__init__()
        self._mixer = mixer
        self._target = target

    def sample1(self, want_value=False) -> ValueType:
        mixer_value = self._mixer.sample1()
        target_frp = self._target(mixer_value)
        target_value = FRP.sample1(target_frp)
        return join_values([mixer_value, target_value])

    def value(self) -> ValueType:
        if self._cached_value is None:
            mixer_value = self._mixer.value()
            target_frp = self._target(mixer_value)
            self._cached_value = join_values([mixer_value, target_frp.value])
        return self._cached_value

    def kind(self) -> Kind:
        if self._cached_kind is None:
            self._cached_kind = self._mixer.kind() >> self._target
        return self._cached_kind

    def clone(self) -> 'MixtureExpression':
        new_expr = MixtureExpression(self._mixer.clone(), self._target.clone())
        new_expr._cached_kind = self._cached_kind
        return new_expr

class ConditionalExpression(FrpExpression):
    def __init__(self, target: FrpExpression, condition: Callable[[ValueType], Any]) -> None:  # Any is morally bool
        super().__init__()
        self._condition = condition   # This will typically be a statistic returning a bool
        self._target = target

    def sample1(self) -> ValueType:
        while True:  # If condition is always false, this will not terminate
            val = self._target.sample1()
            if bool(as_scalar(self._condition(val))):
                return val

    def value(self) -> ValueType:
        # ATTN: Logical oddity about having value() for this type as it is counterfactual
        # Same with computation earlier; it bears thinking about
        # Perhaps this should just be sample1 always?  Or some sort of Bottom
        if self._cached_value is not None:
            return self._cached_value
        val = self._target.value()

        if as_scalar(self._condition(val)):
            return val
        else:
            self._cached_kind = Kind.empty
            return vec_tuple()  # "Value" of Empty FRP

    def kind(self) -> Kind:
        if self._cached_kind is None:
            if self.value() == vec_tuple():
                self._cached_kind = Kind.empty
            else:
                self._cached_kind = self._target.kind() | self._condition
        return self._cached_kind

    def clone(self) -> 'ConditionalExpression':
        new_expr = ConditionalExpression(self._target.clone(), self._condition)
        new_expr._cached_kind = self._cached_kind
        return new_expr

class PureExpression(FrpExpression):
    """An expression representing a specific FRP.

    This acts as a leaf in the expression tree. Note that FRPs are logically
    immutable, and we keep the *specific* FRP as part of the expression. ...ATTN
    """
    def __init__(self, frp: 'FRP') -> None:
        super().__init__()
        self._target = frp
        if frp.is_kinded():
            self._cached_kind = frp.kind
        if frp._value is not None:
            self._cached_value = frp._value

    def sample1(self, want_value=False) -> ValueType:
        return FRP.sample1(self._target)

    def value(self) -> ValueType:
        self._cached_value = self._target.value  # For checks in other expressions
        return self._cached_value

    def kind(self) -> Kind:
        self._cached_kind = self._target.kind    # For checks in other expressions
        return self._cached_kind

    def clone(self) -> 'PureExpression':
        new_expr = PureExpression(self._target.clone())
        new_expr._target._kind = self._target._kind
        return new_expr

    def expectation(self):
        if self._target.is_kinded():
            return self._target.kind.expectation()
        raise ComplexExpectationWarning('The expectation of this FRP could not be computed '
                                        'without first finding its kind.')

def as_expression(frp: 'FRP') -> FrpExpression:
    """Returns an FRP expression that is equivalent to this FRP.

    If kinded, then we merely wrap the FRP itself. However, if the
    FRP is defined by an expression, we reproduce that expression,
    caching the kind and value if they are available.

    """
    if frp.is_kinded():
        return PureExpression(frp)
    assert frp._expr is not None
    return frp._expr


#
# Conditional FRPs
#

class ConditionalFRP:
    """A unified representation of a conditional FRP.

    A conditional FRP is a mapping from a set of values of common
    dimension to FRPs of common dimension. This can be based on
    either a dictionary or on a function, but note that the function
    should return the *same* FRP each time it is called with any
    particular value. (In fact, values are cached here to make it
    easier to define function based condtional FRPs.)

    """
    def __init__(
            self,
            mapping: Callable[[ValueType], 'FRP'] | dict[ValueType, 'FRP'] | ConditionalKind,
            *,
            codim: int | None = None,  # If set to 1, will pass a scalar not a tuple to fn (not dict)
            dim: int | None = None,
            domain: Iterable[ValueType] | None = None
    ) -> None:
        # These are optional hints, useful for checking compatibility (codim=1 is significant though)
        self._codim = codim
        self._dim = dim
        self._domain: set | None = set(domain) if domain else None
        self._is_dict = True
        self._original_fn: Callable[[ValueType], 'FRP'] | None = None

        if isinstance(mapping, ConditionalKind):
            mapping = mapping.map(frp)

        if isinstance(mapping, dict):
            self._mapping: dict[ValueType, 'FRP'] = {as_vec_tuple(k): v for k, v in mapping.items()}

            def fn(*args) -> 'FRP':
                if len(args) == 0:
                    raise MismatchedDomain('A conditional FRP requires an argument, none were passed.')
                if isinstance(args[0], tuple):
                    if self._codim and len(args[0]) != self._codim:
                        raise MismatchedDomain(f'A value of dimension {len(args[0])} passed to a'
                                               ' conditional FRP of codim {self._codim}.')
                    value = VecTuple(args[0])
                elif self._codim and len(args) != self._codim:
                    raise MismatchedDomain(f'A value of dimension {len(args)} passed to a '
                                           'conditional FRP of codim {self._codim}.')
                else:
                    value = VecTuple(args)
                if value not in self._mapping:
                    raise MismatchedDomain(f'Value {value} not in domain of conditional FRP.')
                return self._mapping[value]

            self._fn: Callable[..., 'FRP'] = fn

            if (self._dim is not None and
                any([v.is_kinded() and v.dim != self._dim
                     for _, v in self._mapping.items()])):
                raise ConstructionError('The FRPs produced by a conditional FRP are not all of the same dimension')
        elif callable(mapping):         # Check to please mypy
            self._mapping = {}
            self._is_dict = False
            self._original_fn = mapping

            def fn(*args) -> 'FRP':
                if len(args) == 0:
                    raise MismatchedDomain('A conditional FRP requires an argument, none were passed.')
                if isinstance(args[0], tuple):
                    if self._codim and len(args[0]) != self._codim:
                        raise MismatchedDomain(f'A value of dimension {len(args[0])} passed to a'
                                               f' conditional FRP of mismatched codim {self._codim}.')
                    value = VecTuple(args[0])
                elif self._codim and len(args) != self._codim:
                    raise MismatchedDomain(f'A value of dimension {len(args)} passed to a '
                                           f'conditional FRP of mismatched codim {self._codim}.')
                else:
                    value = VecTuple(args)
                if self._domain and value not in self._domain:
                    raise MismatchedDomain(f'Value {value} not in domain of conditional FRP.')

                if value in self._mapping:
                    return self._mapping[value]
                try:
                    if self._codim == 1:  # pass a scalar
                        result = mapping(value[0])
                    else:
                        result = mapping(value)
                except Exception as e:
                    raise MismatchedDomain(f'encountered a problem passing {value} to a conditional FRP: {str(e)}')
                self._mapping[value] = result   # Cache to ensure we get the same FRP every time with value
                return result

            self._fn = fn

    def __call__(self, *value) -> 'FRP':
        return self._fn(*value)

    def clone(self) -> 'ConditionalFRP':
        if self._is_dict:
            cloned = {k: v.clone() for k, v in self._mapping.items()}
            return ConditionalFRP(cloned, dim=self._dim, codim=self._codim, domain=self._domain)
        else:
            # ATTN! We clone here out of caution, in case a function returns an existing FRP
            # The ConditionalFRP will cache the results for each one, so clone will only
            # be called at most one extra time.

            def fn(value):
                assert self._original_fn is not None
                return self._original_fn(value).clone()

            return ConditionalFRP(fn, dim=self._dim, codim=self._codim, domain=self._domain)

    def expectation(self):
        """Returns a function from values to the expectation of the corresponding FRP.

        Note that for a lazily evaluated FRP, it may be costly to compute the expectation
        so this will fail with a warning. See the forced_expectation and approximate_expectation
        methods for alternatives in that case.

        The domain, dim, and codim of the conditional kind are each included as an
        attribute ('domain', 'dim', and 'codim', respetively) of the returned
        function. These may be None if not available.

        """
        def fn(*x):
            try:
                frp = self._fn(*x)
            except MismatchedDomain:
                return None
            return frp.expectation()

        setattr(fn, 'codim', self._codim)
        setattr(fn, 'dim', self._dim)
        setattr(fn, 'domain', self._domain)

        return fn

    def forced_expectation(self):
        """Returns a function from values to the expectation of the corresponding FRP.

        This forces computation of the expectation even if doing so
        is computationally costly. See expectation and
        approximate_expectation properties for alternatives in that
        case.

        The domain, dim, and codim of the conditional kind are each included as an
        attribute ('domain', 'dim', and 'codim', respetively) of the returned
        function. These may be None if not available.

        """
        def fn(*x):
            try:
                frp = self._fn(*x)
            except MismatchedDomain:
                return None
            return frp.forced_expectation()

        setattr(fn, 'codim', self._codim)
        setattr(fn, 'dim', self._dim)
        setattr(fn, 'domain', self._domain)

        return fn

    def approximate_expectation(self, tolerance=0.01):
        """Returns a function from values to the approximate expectation of the corresponding FRP.

        The approximation is computed to the specified tolerance
        using an appropriate number of samples. See expectation and
        approximate_expectation properties for alternatives in that
        case.

        The domain, dim, and codim of the conditional kind are each included as an
        attribute ('domain', 'dim', and 'codim', respetively) of the returned
        function. These may be None if not available.

        """
        def fn(*x):
            try:
                frp = self._fn(*x)
            except MismatchedDomain:
                return None
            return frp.approximate_expectation(tolerance)

        setattr(fn, 'codim', self._codim)
        setattr(fn, 'dim', self._dim)
        setattr(fn, 'domain', self._domain)

        return fn

    def __str__(self) -> str:
        # if dict put out a table of values and FRP summaries
        # if callable, put out what information we have
        tbl = '\n'.join('  {value:<16s}  {frp:<s}'.format(value=str(k), frp=str(v))
                        for k, v in sorted(self._mapping.items(), key=lambda item: item[0]))
        label = ''
        dlabel = ''
        if self._codim:
            label = label + f' from values of dimension {str(self._codim)}'
        if self._dim:
            label = label + f' to values of dimension {str(self._dim)}'
        if self._domain:
            dlabel = f' with domain={str(self._domain)}'

        if self._is_dict or self._domain == set(self._mapping.keys()):
            return f'A conditional FRP with mapping:\n{tbl}'
        elif tbl:
            cont = '  {value:<16s}  {frp:<s}'.format(value='...', frp='...more FRPs')
            mlabel = f'\nIt\'s mapping includes:\n{tbl}\n{cont}'
            return f'A conditional FRP as a function{dlabel or label or mlabel}'
        else:
            return f'A conditional FRP as a function{dlabel or label}'

    def __frplib_repr__(self):
        if environment.ascii_only:
            return str(self)
        return Panel(str(self), expand=False, box=rich_box.SQUARE)

    def __repr__(self) -> str:
        if environment.is_interactive:
            return str(self)
        label = ''
        if self._codim:
            label = label + f', codim={repr(self._codim)}'
        if self._dim:
            label = label + f', dim={repr(self._dim)}'
        if self._domain:
            label = label + f', domain={repr(self._domain)}'
        if self._is_dict or self._domain == set(self._mapping.keys()):
            return f'ConditionalFRP({repr(self._mapping)}{label})'
        else:
            return f'ConditionalFRP({repr(self._fn)}{label})'

    # FRP operations lifted to Conditional FRPs

    def transform(self, statistic):
        if not isinstance(statistic, Statistic):
            raise FrpError('A conditional FRP can be transformed only by a Statistic.'
                           ' Consider passing this tranform to `conditional_frp` first.')
        lo, hi = statistic.dim
        if self._dim is not None and (self._dim < lo or self._dim > hi):
            raise FrpError(f'Statistic {statistic.name} is incompatible with this FRP: '
                           f'acceptable dimension [{lo},{hi}] but kind dimension {self._dim}.')
        if self._is_dict:
            return ConditionalFRP({k: statistic(v) for k, v in self._mapping.items()})

        if self._dim is not None:
            def transformed(*value):
                return statistic(self._fn(*value))
        else:  # We have not vetted the dimension, so apply with care
            def transformed(*value):
                try:
                    return statistic(self._fn(*value))
                except Exception:
                    raise FrpError(f'Statistic {statistic.name} appears incompatible with this FRP.')

        return ConditionalFRP(transformed)

    def __xor__(self, statistic):
        return self.transform(statistic)

    def __rshift__(self, cfrp):
        if not isinstance(cfrp, ConditionalFRP):
            return NotImplemented
        if self._is_dict:
            return ConditionalFRP({given: frp >> cfrp for given, frp in self._mapping.items()})

        def mixed(*given):
            self(*given) >> cfrp
        return ConditionalKind(mixed)

    def __mul__(self, cfrp):
        if not isinstance(cfrp, ConditionalFRP):
            return NotImplemented
        if self._is_dict and cfrp._is_dict:
            intersecting = self._mapping.keys() & cfrp._mapping.keys()
            return ConditionalFRP({given: self._mapping[given] * cfrp._mapping[given] for given in intersecting})

        def mixed(*given):
            self(*given) * cfrp(*given)
        return ConditionalFRP(mixed)


def conditional_frp(
        mapping: Callable[[ValueType], FRP] | dict[ValueType, FRP] | ConditionalKind | None = None,
        *,
        codim=None,
        dim=None,
        domain=None
) -> ConditionalFRP | Callable[..., ConditionalFRP]:
    """Converts a mapping from values to FRPs into a conditional FRP.

    The mapping can be a dictionary associating values (vector tuples)
    to FRPs, a function associating values to FRPs, or a conditional kind.

    The dictionaries can be specified with scalar keys as these are automatically
    wrapped in a tuple. If you want the function to accept a scalar argument
    rather than a tuple (even 1-dimensional), you should supply codim=1.

    The `codim`, `dim`, and `domain` arguments are used for compatibility
    checks, except for the codim=1 case mentioned earlier. `domain` is the
    set of possible values which can be supplied when mapping is a function
    (or used as a decorator).

    If mapping is missing, this function can acts as a decorator on the
    function definition following.

    Returns a ConditionalFRP (if mapping given) or a decorator.

    """
    if mapping is not None:
        return ConditionalFRP(mapping, codim=codim, dim=dim, domain=domain)

    def decorator(fn: Callable) -> ConditionalFRP:
        return ConditionalFRP(fn, codim=codim, dim=dim, domain=domain)
    return decorator


class EmptyFrpDescriptor:
    def __get__(self, obj, objtype=None):
        return objtype(Kind.empty)


#
# FRPs are logically immutable but the _kind and _expression properties
# are computed lazily and thus can be mutated. If available, only the
# _kind is needed to sample, but if this is too complex, we use the
# _expression steps to generate samples. See FrpExpression and its
# subclasses.
#

class FRP:
    COMPLEXITY_THRESHOLD = 1024  # Maximum size to maintain kindedness

    def __init__(self, create_from: FRP | FrpExpression | Kind | str) -> None:
        if not create_from:  # Kind.empty or FRP.empty or ''
            self._kind: Kind | None = Kind.empty
            self._expr: FrpExpression | None = None
            self._value: ValueType | None = vec_tuple()
            return

        if isinstance(create_from, FRP):  # Like clone, value not copied
            if create_from.is_kinded():
                assert create_from._kind is not None
                create_from = create_from._kind
            else:
                assert create_from._expr is not None
                create_from = create_from._expr

        if isinstance(create_from, FrpExpression):
            self._expr = create_from
            self._kind = create_from._cached_kind    # Computed Lazily
            self._value = create_from._cached_value  # Computed Lazily
        else:
            self._kind = Kind(create_from)
            self._expr = None
            self._value = None

    @classmethod
    def sample(cls, n: int, frp: 'FRP | Kind', summary=True) -> FrpDemoSummary | FrpDemo:
        if isinstance(frp, Kind):
            return _sample_from_kind(n, frp, summary)
        if frp._kind is not None:
            return _sample_from_kind(n, frp._kind, summary)
        assert frp._expr is not None
        return _sample_from_expr(n, frp._expr, summary)

    @classmethod
    def sample1(cls, frp: 'FRP') -> ValueType:
        one_sample = cast(FrpDemo, cls.sample(1, frp, summary=False))
        return as_vec_tuple(one_sample[0])  # ATTN: Wrapping not needed after Quantity conversion

    @property
    def value(self) -> ValueType:
        if self._value is None:
            self._value = VecTuple(self._get_value())  # ATTN: VecTuple not be needed after Quantity conversion
        return self._value

    @property
    def size(self) -> int:
        "Returns the size of the FRP's kind. The kind is computed if not yet available."
        if self._kind is None:
            assert self._expr is not None
            self._kind = self._expr.kind()
        return self._kind.size

    @property
    def dim(self) -> int:
        "Returns the dimension of the FRP's kind. The value is computed if it has not been already."
        if self._kind is None:
            return len(self.value)  # The value is likely cheaper than the kind to produce here
        return self._kind.dim

    @property
    def kind(self) -> Kind:
        if self._kind is None:
            assert self._expr is not None
            self._kind = self._expr.kind()
        return self._kind

    def expectation(self):
        """Returns the expectation of this FRP, unless computationally inadvisable.

        For lazily-computed FRPs, this attempts to compute the expectation *without*
        computing the kind explicitly. This is often but not always possible.
        If computing the kind is required, this raises a ComplexExpectationWarning
        exception.

        To force computation of the expectation in this case, use the
        forced_expectation property. See also the approximate_expectation()
        method, which may be good enough.

        """
        if self.is_kinded():
            return self.kind.expectation()
        else:
            assert self._expr is not None
            return _expectation_from_expr(self._expr)

    def forced_expectation(self):
        "Returns the expectation of this FRP, computing the kind if necessary to do so."
        try:
            return self.expectation()
        except ComplexExpectationWarning:
            return self.kind.expectation()

    def approximate_expectation(self, tolerance=0.01) -> ValueType:
        "Computes an approximation to this FRP's expectation to the specified tolerance."
        n = int(math.ceil(tolerance ** -2))
        return scalarize(sum(FRP.sample(n, self, summary=False)) / as_real(n))  # type: ignore

    empty = EmptyFrpDescriptor()

    def __str__(self) -> str:
        if self._kind == Kind.empty:
            return 'The empty FRP with value <>'
        if self._kind is not None:
            return (f'An FRP with value {show_tuple(self.value, max_denom=10)}')
        return f'An FRP with value {show_tuple(self.value, max_denom=10)}. (It may be slow to evaluate its kind.)'

    def __frplib_repr__(self) -> str:
        if self._kind == Kind.empty:
            return ('The [bold]empty FRP[/] of dimension [#3333cc]0[/] with value [bold #4682b4]<>[/]')
        if self._kind is not None:
            return (f'An [bold]FRP[/] with value [bold #4682b4]{self.value}[/]')
        return f'An [bold]FRP[/] with value [bold #4682b4]{self.value}[/]. (It may be slow to evaluate its kind.)'

    def __repr__(self) -> str:
        if environment.is_interactive:
            return f'FRP(value={show_tuple(self.value, max_denom=10)})'
        return super().__repr__()

    def __bool__(self) -> bool:
        return self.dim > 0

    def __iter__(self):
        yield from self.value

    def _get_value(self):
        "Like FRP.sample1 but gets actual value from an expression. Only call if _value is None."
        if self._kind is not None:
            return FRP.sample1(self)
        assert self._expr is not None
        return self._expr.value()

    def is_kinded(self):
        "Returns true if this FRP has an efficiently available kind."
        return self._expr is None and self._kind is not None

    def clone(self) -> FRP:
        if self.is_kinded():
            new_frp = FRP(self.kind)
        else:
            assert self._expr is not None   # Grrr...
            new_frp = FRP(self._expr.clone())
            new_frp._kind = self._kind  # If already computed, use it.
        return new_frp

    # Operations and Operators on FRPs that mirrors the same for Kinds
    # These all produce new FRPs. We use an expression for the FRPs
    # because these operations relate both the kinds *and* the values.
    # ATTN: when a kind is useful but not demand, we will compute
    # the kind if the complexity is below a threshold.

    def independent_mixture(self, frp: 'FRP') -> 'FRP':
        we_are_kinded = self.is_kinded() and frp.is_kinded()

        if self._value is not None and frp._value is not None:
            value = join_values([self._value, frp._value])
        else:
            value = None

        # Redundant assertions here because mypy doesn't know is_kinded => ._kind is not None
        if we_are_kinded and self._kind is not None and frp._kind is not None and \
           self._kind.size * frp._kind.size <= self.COMPLEXITY_THRESHOLD:
            spec = self._kind * frp._kind
            value = value or join_values([self.value, frp.value])  # Generate values if needed
        elif isinstance(self._expr, IMixtureExpression) and isinstance(frp._expr, IMixtureExpression):
            spec = IMixtureExpression.join(self._expr, frp._expr)
        elif isinstance(self._expr, IMixtureExpression):
            spec = IMixtureExpression.append(self._expr, as_expression(frp))
        elif isinstance(frp._expr, IMixtureExpression):
            spec = IMixtureExpression.prepend(frp._expr, as_expression(self))
        else:
            spec = IMixtureExpression([as_expression(self), as_expression(frp)])

        result = FRP(spec)
        if value is not None:
            result._value = value
        return result

    def __mul__(self, other):   # Self -> FRP -> FRP
        "Mixes FRP with another independently"
        return self.independent_mixture(other)

    def __pow__(self, n, modulo=None):  # Self -> int -> FRP
        is_kinded = self.is_kinded()
        if is_kinded and math.log2(self.size) * n <= math.log2(self.COMPLEXITY_THRESHOLD):
            return FRP(self.kind ** n)

        if not is_kinded and isinstance(self._expr, IMixPowerExpression):
            expr = IMixPowerExpression(self._expr._term, self._expr._pow + n)
        else:
            expr = IMixPowerExpression(as_expression(self), n)
        return FRP(expr)

    def __rshift__(self, c_frp):
        "Mixes this FRP with FRPs given for each value"
        c_frp = conditional_frp(c_frp)  # Convert to proper form with a copy  ATTN: clone this?
        resolved = False
        if self.is_kinded():
            assert self._kind is not None
            dim = self._kind.dim
            targets = {branch.vs: c_frp(branch.vs) for branch in self._kind._branches}
            viable = True
            for target in targets.values():
                if not target.is_kinded() or (dim * target._kind.dim > self.COMPLEXITY_THRESHOLD):
                    viable = False
                    break
            if viable:   # Return a kinded FRP
                c_kind = {k: frp.kind for k, frp in targets.items()}
                result = FRP(self._kind >> c_kind)
                # Since we're kinded, generate the values now
                result._value = join_values([self.value, c_frp(self.value).value])
                resolved = True
            else:
                resolved = False
        if not resolved:
            expr = MixtureExpression(as_expression(self), c_frp)
            result = FRP(expr)
            if self._value is not None:
                frp = c_frp(self._value)
                if frp._value is not None:
                    result._value = join_values([self._value, frp._value])
        return result

    def transform(self, f_mapping):
        "Applies a transform/Statistic to an FRP"
        # ATTN! Handle actual statistic here; error checking etc.
        if isinstance(f_mapping, Statistic):
            if self.is_kinded():
                fdim_lo, fdim_hi = f_mapping.dim
                if self.dim < fdim_lo or self.dim > fdim_hi:
                    raise MismatchedDomain(f'Statistic {f_mapping.name} is incompatible with this FRP: '
                                           f'acceptable dimension [{fdim_lo},{fdim_hi}] but FRP dimension {self.dim}.')
            elif self._value is not None:  # Should we just get the value here or risk an error? ATTN
                try:
                    f_mapping(self._value)
                except Exception:
                    raise MismatchedDomain(f'Statistic {f_mapping.name} is incompatible with this FRP: '
                                           f'could not evaluate it on the FRPs value.')
            stat: Callable = f_mapping
        else:
            stat = tuple_safe(f_mapping)
        if self.is_kinded():
            assert self._kind is not None
            result = FRP(self._kind ^ stat)
            result._value = stat(self.value)
        else:
            result = FRP(TransformExpression(stat, as_expression(self)))
            # ATTN: this next might be handled by caching!
            if self._value is not None:
                result._value = stat(self._value)
        return result

    def __xor__(self, f_mapping):
        "Applies a transform/Statistic to an FRP"
        return self.transform(f_mapping)

    def __rfloordiv__(self, c_frp):
        "Conditioning on self; other is a conditional FRP."
        c_frp = conditional_frp(c_frp)  # Convert to proper form with a copy
        return c_frp(self.value)   # generate value even if costly

    @overload
    def marginal(self, *__indices: int) -> 'FRP':
        ...

    @overload
    def marginal(self, __subspace: Iterable[int] | Projection | slice) -> 'FRP':
        ...

    def marginal(self, *index_spec) -> 'FRP':
        dim = self.dim

        # Unify inputs
        if len(index_spec) == 0:
            return FRP.empty
        if isinstance(index_spec[0], Iterable):
            indices: tuple[int, ...] = tuple(index_spec[0])
        elif isinstance(index_spec[0], Projection):
            indices = tuple(index_spec[0].subspace)
        elif isinstance(index_spec[0], slice):
            start, stop, step = index_spec[0].indices(dim + 1)
            indices = tuple(range(max(start, 1), stop, step))
        else:
            indices = index_spec

        if len(indices) == 0:
            return FRP.empty

        # Check dimensions (allow negative indices python style)
        if any([index == 0 or index < -dim or index > dim for index in indices]):
            raise FrpError( f'All marginalization indices in {indices} should be between 1..{dim} or -{dim}..-1')

        # Marginalize
        def marginalize(value):
            return as_vec_tuple(map(lambda i: value[i - 1] if i > 0 else value[i], indices))

        if self.is_kinded():
            stat = Statistic(marginalize, dim=0, codim=len(indices))
            return stat(self)

        assert self._expr is not None
        expr = TransformExpression(marginalize, self._expr)
        if self._value is not None:
            expr._cached_value = marginalize(self._value)
        if self._kind is not None:
            expr._cached_kind = self._kind.map(marginalize)

        return FRP(expr)

    def __getitem__(self, indices):
        "Marginalizing this kind; other is a projection index or list of indices (1-indexed)"
        return self.marginal(indices)

    def __or__(self, predicate):
        "Applies a conditional filter to an FRP"
        if isinstance(predicate, Statistic):
            condition: Callable = predicate
        elif callable(predicate):
            condition = tuple_safe(predicate)   # ATTN: update?
        else:
            raise ConditionMustBeCallable('A conditional requires a condition after the given bar.')

        # ATTN: Can avoid evaluating the value until asked by using the expression version
        # But the cost here does not seem high in the kinded case, so we can activate it.
        # And logically that makes sense; else how do we think about the conditional.

        if self.is_kinded():
            relevant = condition(self.value)  # We evaluate the value here
            if not relevant:
                return FRP.empty
            conditional = FRP(self.kind | condition)
            conditional._value = self._value
            return conditional

        conditional = FRP(ConditionalExpression(as_expression(self), condition))
        if self._kind:
            conditional._kind = self._kind | condition

        if self._value is not None and condition(self._value):
            conditional._value = self._value

        return conditional

    def __rmatmul__(self, statistic):
        "Returns a transformed FRP with the original FRP as context for conditionals."
        if isinstance(statistic, Statistic):
            return TaggedFRP(self, statistic)
        return NotImplemented

#
# Constructors
#

def frp(spec) -> FRP:
    """A generic constructor for FRPs from a variety of objects.

    Parameter `spec` can be a string, a Kind, another FRP, or an FRP
    expression.

    Returns a fresh FRP corresponding to spec. If the input is an
    FRP, this returns a clone.

    """
    if isinstance(spec, str):
        return FRP(kind(spec))

    try:
        return FRP(spec)
    except Exception as e:
        raise FrpError(f'I could not create an Frp from {spec}: {str(e)}')


#
# Tagged FRPs for context in conditionals
#
# phi@X acts exactly like phi(X) except in a conditional, where
#    phi@X | (s(X) == v)
# is like
#    (X * phi(X) | (s(Proj[:(d+1)](__)) == v))[(d+1):]
# but simpler
#

class TaggedFRP(FRP):
    def __init__(self, createFrom, stat: Statistic):
        original = frp(createFrom)
        super().__init__(original.transform(stat))
        self._original = original
        self._stat = stat

        lo, hi = stat.dim
        if self.dim < lo or self.dim > hi:
            raise MismatchedDomain(f'Statistic {stat.name} is incompatible with this Kind, '
                                   f'which has dimension {self.dim} out of expected range '
                                   f'[{lo}, {"infinity" if hi == infinity else hi}].')

    def __or__(self, condition):
        return self._original.__or__(condition).transform(self._stat)

    def transform(self, statistic):
        # maybe some checks here
        new_stat = compose2(statistic, self._stat)
        return TaggedFRP(self._original, new_stat)

    def _untagged(self):
        return (self._stat, self._original)


#
# Utilities and Additional Factories and Combinators
#

class FisherYates(FrpExpression):
    def __init__(self, items: Iterable):
        super().__init__()
        self.items = tuple(items)
        self.n = len(self.items)

    def sample1(self):
        permuted = list(self.items)
        for i in range(self.n - 1):
            j = random.randrange(i, self.n)
            permuted[j], permuted[i] = permuted[i], permuted[j]  # swap
        return VecTuple(permuted)

    def value(self):
        if self._cached_value is None:
            self._cached_value = self.sample1()
        return self._cached_value

    def kind(self) -> Kind:
        if self.n <= 10:
            return permutations_of(self.items)
        raise KindError(f'The kind of a large ({self.n} > 10) permutation is too costly to compute.')

    def clone(self) -> 'FisherYates':
        self._cached_value = None
        self.value()
        return self

def shuffle(items: Iterable) -> FRP:
    return frp(FisherYates(items))


#
# Low-level Helpers
#

def _sample_from_kind(n: int, kind: Kind, summary: bool) -> FrpDemoSummary | FrpDemo:
    if summary:
        table = FrpDemoSummary()
        for _ in range(n):
            table.add(VecTuple(kind.sample1()))  # ATTN: VecTuple wrapping unneeded after Quantity conversion
        return table
    return kind.sample(n)   # ATTN: should VecTuple wrap here for now

def _sample_from_expr(n: int, expr: FrpExpression, summary: bool) -> FrpDemoSummary | FrpDemo:
    if summary:
        table = FrpDemoSummary()
        for _ in range(n):
            table.add(VecTuple(expr.sample1()))  # ATTN: VecTuple wrapping unneeded after Quantity conversion
        return table
    values = []
    for _ in range(n):
        values.append(VecTuple(expr.sample1()))
    return values

def _expectation_from_expr(expr: FrpExpression):
    # ATTN: Expand the range of things that this works for
    # For instance, mixture powers or PureExpressions where the kind is available
    # should be automatic
    if expr._cached_kind is not None:
        return expr._cached_kind.expectation()
    raise ComplexExpectationWarning('The expectation of this FRP could not be computed without first finding its kind.')

def join_values(values: Iterable[ValueType]) -> ValueType:
    combined = []
    for value in values:
        combined.extend(list(value))
    return VecTuple(combined)


#
# Info tags
#

setattr(frp, '__info__', 'frp-factories')
setattr(conditional_frp, '__info__', 'frp-factories')
setattr(shuffle, '__info__', 'frp-factories')
