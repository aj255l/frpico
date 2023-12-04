from frp_pico import Kind

def uniform(values):
    '''
    Returns a Kind with all values weighted equally.
    '''
    return Kind(zip(values, [1] * len(values)))

def constant(value):
    '''
    Returns a Kind with one value.
    '''
    return Kind([(value, 1)])

def weighted(values, weights):
    '''
    Returns a custom Kind with specified values and weights.
    '''
    return Kind(zip(values, weights))