from __future__        import annotations

from collections       import defaultdict
from functools         import reduce
from typing_extensions import Any

from parsy import (
    ParseError,
    regex,
    seq,
    string,
    whitespace,
)

from frplib_pico.exceptions           import KindError
from frplib_pico.numeric              import as_numeric, integer_re, numeric_q_from_str, numeric_re
from frplib_pico.vec_tuples           import VecTuple
from frplib_pico.parsing.parsy_adjust import (generate, parse_error_message,
                                         repeat_within, with_desc, with_label)

# ATTN: Use numeric.Numeric types here for the tree.

#
# Helpers
#

def convert(xs: tuple, f) -> VecTuple:
    return VecTuple(f(x) for x in xs)


#
# Basic Combinators
#

numeric_p = regex(numeric_re).map(numeric_q_from_str)
integer_p = regex(integer_re).map(numeric_q_from_str)

valuei = with_label('a numeric component of a value', numeric_p)
ws = with_label('whitespace', whitespace).optional()
otuple = string('<')
ctuple = string('>')
oparen = string('(')
cparen = string(')')
comma_ws = seq(string(","), ws)

weight = with_label('a branch weight', numeric_p)  # Allow sign so that we can flag the error more clearly
value_tuple = with_desc(
    'a node value',
    seq(ws, otuple, ws) >> valuei.sep_by(comma_ws) << seq(ws, ctuple, ws)
).map(VecTuple)


#
# Main Parser
#

@generate  # ('a kind represented as a string')
def kind_sexp():
    yield oparen
    yield ws
    root = yield value_tuple
    yield ws
    children = yield repeat_within(
        None,
        seq(weight << ws, (kind_sexp | value_tuple) << ws),
        cparen,
        labels={'close': ')', 'item': 'a kind subtree/leaf value'}
    )
    # Need to check dimensions of tuples inside, consistent values, etc.
    result = [root]
    result.extend(children)
    return result

def parse_kind_sexp(s: str, rich=True, short=False) -> list:
    try:
        ktree = kind_sexp.parse(s)
    except ParseError as e:
        raise KindError(parse_error_message(e, rich, short))

    perrors = validate_kind(ktree, sep=", ")
    if perrors:
        raise KindError(f'Errors found in kind string format: {perrors}.')
    return ktree

#
# Validation
#

# Examples:
# [(), [1, [(2,), [3, (2, 4)], [5, (2, 6)]]], [1, [(7,), [8, (7, 9)], [10, (7, 12)]]]]
# [(), [1, (2,)], [3, (4,)]]
# [(), [1, (2, 3)], [4, [(5,), [1, (5, 7)]]]]   # Unusual but OK???
#
# Need:
# 1. All paths consistent:  prefix(node) == parent
# 2. Values added within each subtree are distinct
# 3. All leaves have the same dimension

def check_kind_tree(tree, errors, min_leaf_dim=None, max_leaf_dim=None):
    parent, *branches = tree
    p_str = convert(parent, str)

    def dm(x, v, f):
        if v is None:
            return x
        return f(v, x)

    if len(branches) == 0:
        # Handle root separately, this is an error or just remove these earlier
        return (errors, min_leaf_dim, max_leaf_dim)

    next_added: dict[Any, int] = defaultdict(int)
    for branch in branches:
        weight, subtree = branch
        if weight <= 0:
            errors.append(f'Non-positive weight {weight} at subtree {parent}.')
        if isinstance(subtree, tuple):  # Leaf case
            if len(subtree) <= len(parent):
                st_str = convert(subtree, str)
                errors.append(f'Node {st_str} does not extend {p_str}')
            elif subtree[0:len(parent)] != parent:
                st_str = convert(subtree, str)
                errors.append(f'Node {st_str} inconsistent with parent {p_str}')
            else:
                next_added[subtree[len(parent):]] += 1
                min_leaf_dim = dm(len(subtree), min_leaf_dim, min)
                max_leaf_dim = dm(len(subtree), max_leaf_dim, max)
        else:  # Subtree
            if len(subtree[0]) <= len(parent):
                st_str = convert(subtree[0], str)
                errors.append(f'Node {st_str} does not extend its parent {p_str}')
            elif subtree[0][0:len(parent)] != parent:
                st_str = convert(subtree[0], str)
                errors.append(f'Node {st_str} at subtree inconsistent with parent {p_str}')
            else:
                next_added[subtree[0][len(parent):]] += 1
                errors, min_leaf_dim, max_leaf_dim = check_kind_tree(subtree, errors, min_leaf_dim, max_leaf_dim)

    for k, v in next_added.items():
        if v > 1:
            errors.append(f'Distinct branches have a common prefix '
                          f'<{", ".join([str(x) for x in [*p_str, *k]])}>'
                          f' in subtree directly below {p_str}.')

    return (errors, min_leaf_dim, max_leaf_dim)

def validate_kind(kind_tree, sep="\n"):
    errors, min_dim, max_dim = check_kind_tree(kind_tree, [])
    if min_dim != max_dim:
        errors.append(f'Leaf values in the tree have differing dimensions ({min_dim} < {max_dim}).')
    return sep.join(errors)


#
# Canonical Form
#

def canonical_tree(tree, p=1, weight_fn=as_numeric, value_fn=as_numeric):
    # Assumes tree has been validated already
    parent, *branches = tree

    if len(branches) == 0:
        return [p, convert(parent, value_fn)]

    level_sum = reduce(lambda acc, x: acc + weight_fn(x[0]), branches, 0)
    normalized = [[p * weight_fn(b[0]) / level_sum, b[1]] for b in branches]

    canonical = []
    for branch in normalized:
        weight, subtree = branch
        if isinstance(subtree, tuple):
            canonical.append([weight, convert(subtree, value_fn)])
        else:
            canonical.extend(canonical_tree(subtree, weight, weight_fn, value_fn))
    return canonical

# def canonical_tree(tree, p=1):
#     # Assumes tree has been validated already
#     parent, *branches = tree
#
#     if len(branches) == 0:
#         return [p, parent]
#
#     level_sum = reduce(lambda acc, x: acc + x[0], branches, 0)
#     normalized = [[p * b[0] / level_sum, b[1]] for b in branches]
#
#     canonical = []
#     for branch in normalized:
#         weight, subtree = branch
#         if isinstance(subtree, tuple):
#             canonical.append(branch)
#         else:
#             canonical.extend(canonical_tree(subtree, weight))
#     return canonical
