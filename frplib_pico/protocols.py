from __future__ import annotations

from typing            import (
    Protocol,
    runtime_checkable
)

from collections.abc   import Collection


@runtime_checkable
class Projection(Protocol):
    @property
    def subspace(self) -> Collection[int]:
        ...

## ATTN! add *a, **kw to this signature for utility purposes
@runtime_checkable
class Renderable(Protocol):
    def __frplib_repr__(self):
        ...

@runtime_checkable
class Transformable(Protocol):
    def transform(self, f_mapping):
        ...

@runtime_checkable
class SupportsExpectation(Protocol):
    def expectation(self):
        ...

@runtime_checkable
class SupportsApproxExpectation(Protocol):
    def approximate_expectation(self, tolerance):
        ...

@runtime_checkable
class SupportsForcedExpectation(Protocol):
    def forced_expectation(self):
        ...
