# output.py - tools for managing terminal output

from __future__ import annotations

from dataclasses       import dataclass
from decimal           import Decimal
from typing            import Literal
from typing_extensions import Any

from rich              import box
from rich.panel        import Panel

from frplib_pico.env        import environment
from frplib_pico.numeric    import as_nice_numeric

#
# Rendered Output
#

def in_panel(
        s: str,
        box=box.SQUARE,
        title: str | None = None,
        title_align: Literal['left', 'center', 'right'] = 'center',
        subtitle: str | None = None,
        subtitle_align: Literal['left', 'center', 'right'] = 'center',
) -> str | Panel:
    if environment.ascii_only:
        return s
    return Panel(
        s,
        expand=False,
        box=box,
        title=title,
        title_align=title_align,
        subtitle=subtitle,
        subtitle_align=subtitle_align,
    )


#
# Wrapped Quantities Providing Rich String Representations
#

@dataclass(frozen=True)
class RichQuantity:
    this: Any

    def __str__(self) -> str:
        return str(self.this)

    def __repr__(self) -> str:
        return repr(self.this)

    def __frplib_repr__(self):
        if environment.ascii_only:
            return str(self)
        return Panel(str(self), expand=False, box=box.SQUARE)

@dataclass(frozen=True)
class TitledRichQuantity:
    this: Any
    title: str = ''

    def __str__(self) -> str:
        return self.title + str(self.this)

    def __repr__(self) -> str:
        return repr(self.this)

    def __frplib_repr__(self):
        if environment.ascii_only:
            return str(self)
        return Panel(str(self), expand=False, box=box.SQUARE)

@dataclass(frozen=True)
class RichFacade:
    this: Any
    facade: str = ''

    def __str__(self) -> str:
        if self.facade:
            return self.facade
        else:
            return str(self.this)

    def __repr__(self) -> str:
        return repr(self.this)

    def __frplib_repr__(self):
        if environment.ascii_only:
            return str(self)
        return Panel(str(self), expand=False, box=box.SQUARE)

@dataclass(frozen=True)
class TitledRichFacade:
    this: Any
    facade: str = ''
    title: str = ''

    def __str__(self) -> str:
        if self.facade:
            return self.title + self.facade
        else:
            return self.title + str(self.this)

    def __repr__(self) -> str:
        return repr(self.this)

    def __frplib_repr__(self):
        if environment.ascii_only:
            return str(self)
        return Panel(str(self), expand=False, box=box.SQUARE)

class RichString(str):
    def __frplib_repr__(self):
        return str(self)

class RichReal(Decimal):
    def __frplib_repr__(self):
        return str(as_nice_numeric(self))
