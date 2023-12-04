from __future__ import annotations

from parsy     import (
    Parser,
    ParseError,
    Result,
)
from collections.abc   import Iterable
from functools         import wraps
from typing_extensions import Any


#
# Helpers
#

def join_nl(terms: Iterable[str], *, sep: str = ", ", last_sep=', or ', prefixes=['', '', '']) -> str:
    terms = list(terms)
    count = len(terms)
    if count == 1:
        joined = prefixes[0] + terms[0]
    elif count == 2:
        joined = prefixes[1] + f'{terms[0]} or {terms[1]}'
    else:
        joined = prefixes[2] + sep.join(terms[0:-1]) + last_sep + terms[-1]
    return joined


def join_expecteds(expecteds: frozenset[str]):
    return join_nl(list(expecteds), prefixes=['', 'either ', 'one of '])


#
# Fix descriptions and aggregation of results
#

def supplant_result(new_result: Result, base_result: Result | None) -> Result:
    return new_result

def combine_result(new_result: Result, base_result: Result | None) -> Result:
    return new_result.aggregate(base_result)

def extend_result(new_result: Result, base_result: Result | None) -> Result:
    if not base_result:
        return new_result

    if new_result.furthest > base_result.furthest:
        return new_result
    else:
        new_exps = join_expecteds(new_result.expected)
        base_exps = join_expecteds(base_result.expected)
        return Result(
            new_result.status and base_result.status,
            min(new_result.index, base_result.index),
            new_result.value,
            base_result.furthest,
            frozenset([f'{new_exps} after seeing {base_exps}'])
        )

def wrap_result(new_result: Result, base_result: Result | None) -> Result:
    if not base_result:
        return new_result

    if new_result.furthest > base_result.furthest:
        return new_result
    else:
        new_exps = join_expecteds(new_result.expected)
        base_exps = join_expecteds(base_result.expected)
        return Result(
            new_result.status and base_result.status,
            min(new_result.index, base_result.index),
            new_result.value and base_result.value,
            base_result.furthest,
            frozenset([f'{new_exps}, looking for {base_exps}'])
        )

def with_desc(description: str, p: Parser, aggregate=combine_result) -> Parser:
    def p_with_desc(stream, index) -> Result:
        result = p(stream, index)
        if result.status:
            return result
        else:
            return aggregate(Result.failure(index, description), result)
    return Parser(p_with_desc)

def with_label(description: str, p: Parser) -> Parser:
    return with_desc(description, p, aggregate=supplant_result)

# Fixes bug #77 in parsy Parser.desc()
def generate(fn) -> Parser:
    """
    Creates a parser from a generator function
    """
    if isinstance(fn, str):
        return lambda f: with_desc(fn, generate(f))

    @Parser
    @wraps(fn)
    def generated(stream, index):
        # start up the generator
        iterator = fn()

        result = None
        value = None
        try:
            while True:
                next_parser = iterator.send(value)
                result = next_parser(stream, index).aggregate(result)
                if not result.status:
                    return result
                value = result.value
                index = result.index
        except StopIteration as stop:
            returnVal = stop.value
            if isinstance(returnVal, Parser):
                return returnVal(stream, index).aggregate(result)

            return Result.success(index, returnVal).aggregate(result)

    return generated

# Fixes expecteds behavior of builtin Parser method
def bind(parser, bind_fn):
    @Parser
    def bound_parser(stream, index):
        result = parser(stream, index)
        if result.status:
            next_parser = bind_fn(result.value)
            next_result = next_parser(stream, result.index)
            print('>>= ', next_parser, result, next_result)
            return extend_result(next_result, result)
        else:
            return result

    return bound_parser
#
# More functional version of util with better messages
#

def repeat_within(open_delim: Parser | None, item: Parser, close_delim: Parser,
                  min: int = 0, max: int = 1000000000000,  # float("inf"),
                  labels: dict[str, str] = {}) -> Parser:
    """
    ATTN
    Returns a parser that expects the initial parser followed by ``other``.
    The initial parser is expected at least ``min`` times and at most ``max`` times.
    By default, it does not consume ``other`` and it produces a list of the
    results excluding ``other``. If ``consume_other`` is ``True`` then
    ``other`` is consumed and its result is included in the list of results.
    """

    @Parser
    def within_parser(stream, index):
        values: list[Any] = []
        times = 0
        default_labels = {
            'open': 'opening delimiter',
            'close': 'closing delimiter',
            'item': 'item'
        }
        full_labels = default_labels | labels

        if open_delim is not None:
            res = open_delim(stream, index)
            if not res.status:
                Result.failure(index, full_labels['open'])

        while True:

            # try parser first
            res = close_delim(stream, index)
            if res.status and times >= min:
                return Result.success(res.index, values)

            # exceeded max?
            if times >= max:
                # return failure, it matched parser more than max times
                return Result.failure(index, f"{full_labels['item']} at most {max} times")

            # looking for an item
            result = item(stream, index)
            if result.status:
                # consume
                values.append(result.value)
                index = result.index
                times += 1
            elif times >= min:
                # return failure, parser is not followed by other
                return Result.failure(index, full_labels['item']).aggregate(result)  # ATTN
            else:
                # return failure, it did not match parser at least min times
                return Result.failure(index, f"{full_labels['item']} at least {min} times, saw {times}")

    return within_parser


#
# Handling Parsy ParseError
#
# ATTN: Want to handle rich/formatted_text as mentioned here

def parse_error_message(e: ParseError, rich=True, short=False) -> str:  # FormattedText:
    # short implies not rich; i.e., only check if rich is false
    start_context = max(e.index - 5, 0)
    end_context = min(e.index + 6, len(e.stream))
    joined = join_expecteds(e.expected)
    cont = '...' if e.index > 8 else ''
    pad = 3 if e.index > 8 else 0

    # Rich text here: gray for good context, red for bad, and bold red for the pointer
    parsed = cont + e.stream[start_context:e.index]  # style='class:parse.parsed'
    error = e.stream[e.index]  # style='class:parse.error')
    rest = e.stream[(e.index + 1):end_context]
    indent = '     ' + ' ' * (e.index - start_context + pad)
    mark = '^'  # style='class:parse.error')

    mesg = f'I expected to see {joined} at character {e.index + 1}:\n'
    # text = '    "' + parsed + error + e.stream[(e.index + 1):end_context] + '"'
    # pntr = '     ' + (' ' * (e.index - start_context + pad)) + mark
    #
    # return f'{mesg}\n{text}\n{pntr}'

    # return FormattedText([
    #     ('', mesg),
    #     ('', '    "'),
    #     ('class:parse.parsed', parsed),
    #     ('class:parse.error', error),
    #     ('', e.stream[(e.index + 1):end_context] + '"\n'),
    #     ('', '     ' + (' ' * (e.index - start_context + pad))),
    #     ('class:parse.error', mark),
    # ])

    if rich:  # Using rich styles
        return f'{mesg}    "[#71716f]{parsed}[/][#ff0f0f bold]{error}[/]{rest}"\n{indent}[#ff0f0f bold]{mark}[/]'
    elif short:
        return (f'Expected {joined} at character {e.index + 1}: '
                f'"{parsed}*{error}{rest}"')
    else:
        return f'{mesg}    "{parsed}{error}{rest}"\n{indent}{mark}'
