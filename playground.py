# from frplib_pico.exceptions import *
from frplib_pico.kinds import Kind, weighted_as, constant

'''
Returns the kind as a string representation
'''
def displayKind(kind: Kind):
    return kind.show_full()

'''
Observes the FRP and returns the observed value.
Returns a scalar value if the result is of length 1.
'''
def observeFRP(kind: Kind):
    result = kind.sample1()
    if len(result) > 1:
        return str(result)
    return str(result[0])

k = weighted_as([0, 1], weights=[0.5, 0.5])

'''
Takes in a list of (value, weight) pairs and returns
a Kind with those values and weights.
'''
def customKind(branches):
    values, weights = zip(*branches)
    return weighted_as(values, weights=weights)

'''

'''
def constantKind(v):
    return constant(v)