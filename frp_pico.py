import random

# Adapted from of https://github.com/genovese/frplib.
# An incomplete subset of frplib built for the FRPico.

class Kind:
    def normalize(self, branches):
        '''
        Assumes integer weights and integer or float values.
        Branches are assumed to be in the order (value, weight).
        '''
        mapping = dict()
        total = 0
        for (value, weight) in branches:
            mapping[value] = mapping.get(value, 0) + weight
            total += weight

        # TODO: Re-normalize with LCD of fraction list

        return list(sorted(mapping.items(), key=lambda x : x[0])), total

    def __init__(self, branches):
        # Normalise and create value-weight mapping
        self.branches, self.total = self.normalize(branches)
        self.values, self.weights = zip(*self.branches)

        # Variables for displaying the kind
        self.row = 0
        self.col = 0
        self.rows = len(self.values)
        self.cols = max(map(len, self.display().splitlines()))

    def sample(self):
        '''
        Returns a scalar, since all values are assumed to be scalar.
        '''
        r = random.uniform(0, self.total)
        total = 0
        for (value, weight) in self.branches:
            total += weight
            if r <= total:
                return value
            
    def display(self, prepend=''):
        '''
        Returns an ascii representation of the kind.
        Taken from frplib (linked in header).
        Also takes in an addition argument to prepend to the root, if available.
        '''
        if len(self.branches) == 0:
            return '<> -+'

        size = self.rows
        juncture, extra = (size // 2, size % 2 == 0)

        pLabels = list(map(lambda x : f'{x}/{self.total}', self.weights))
        vLabels = list(map(str, self.values))
        pwidth = max(map(len, pLabels), default=0) + 2
        prePad = ' ' * len(prepend)

        lines = []
        if size == 1:
            plab = ' ' + pLabels[0] + ' '
            vlab = vLabels[0].replace(', -', ',-')  # ATTN:HACK fix elsewhere, e.g., '{0:-< }'.format(Decimal(-16.23))
            lines.append(f'{prepend}-{plab:-<{pwidth}}- {vlab}')
        else:
            for i in range(size):
                plab = ' ' + pLabels[i] + ' '
                vlab = vLabels[i].replace(', -', ',-')   # ATTN:HACK fix elsewhere
                if i == 0:
                    lines.append(f'{prePad} ,-{plab:-<{pwidth}}- {vlab}')
                    if size == 2:
                        lines.append(f'{prepend}-|')
                elif i == size - 1:
                    lines.append(f'{prePad} `-{plab:-<{pwidth}}- {vlab}')
                elif i == juncture:
                    if extra:
                        lines.append(f'{prepend}-|')
                        lines.append(f'{prePad} |-{plab:-<{pwidth}}- {vlab}')
                    else:
                        lines.append(f'{prepend}-+-{plab:-<{pwidth}}- {vlab}')
                else:
                    lines.append(f'{prePad} |-{plab:-<{pwidth}}- {vlab}')
        return '\n'.join(lines)

    #####################
    # Scrolling
    #####################
    def scrollDown(self):
        self.row = min(self.row + 1, self.rows)

    def scrollUp(self):
        self.row = max(0, self.row - 1)

    def scrollLeft(self):
        self.col = max(0, self.col - 1)
    
    def scrollRight(self):
        self.col = min(self.col + 1, self.cols - 1)

class ConditionalKind(Kind):
    def __init__(self, mapping):
        # Assume mapping is a dictionary from values to Kinds
        self.mapping = mapping

        # Variables for displaying the kind
        self.row = 0
        self.col = 0
        self.rows = len(self.display().splitlines())
        self.cols = max(map(len, self.display().splitlines()))

    def display(self):
        vWidth = max(map(len, map(str, self.mapping.keys())))

        lines = []
        for value in sorted(self.mapping.keys()):
            kind = self.mapping[value]
            lines.append(kind.display(prepend=f'{value}: '.ljust(vWidth)))

        return '\n\n'.join(lines)

class FRP:
    def __init__(self, kind: Kind):
        self.kind = kind

        # Observed value: None if unobserved, otherwise an int or float.
        self.observed = None

    def observe(self):
        '''
        Observes the FRP. Returns None.
        '''
        self.observed = self.kind.sample()

    def getObserved(self):
        '''
        Returns a scalar value if the result is of length 1.
        '''
        return self.observed

    def isObserved(self):
        return self.observed is not None

    def display(self):
        return self.kind.display()

    def reset(self):
        self.observed = None

class ConditionalFRP(FRP):
    def __init__(self, kind: ConditionalKind):
        super().__init__(kind)

        # Given value: None if no value is given, otherwise the Kind mapped to kind.mapping
        self.given = None

    def observe(self):
        '''
        Observes the conditional FRP if a value has been given.
        '''
        if self.given is not None:
            self.observed = self.given.sample()

    def isGiven(self):
        return self.given is not None

    def giveObserved(self, value):
        '''
        Gives an observed value to the Conditional Kind, returning the appropriate Kind.
        '''
        if value not in self.kind.mapping:
            print(f"ERROR: {value} is not in the conditional kind!!")
            return
        self.given = self.kind.mapping[value]

    def display(self):
        if self.given is None:
            return self.kind.display()
        return self.given.display()

    def reset(self):
        self.observed = None
        self.given = None