class FRP:
    def __init__(self, kind: Kind):
        # Initialize the kind of the FRP
        self.kind = kind
        self.rows = kind.size
        self.cols = max(map(len, kind.show_full().splitlines()))

        # Variables for displaying the kind
        self.kindRow = 0
        self.kindCol = 0

        # Observed value: None if unobserved, otherwise a VecTuple
        self.observed = None

    def observe(self):
        '''
        Observes the FRP. Returns None.
        '''
        self.observed = self.kind.sample1()

    def getObserved(self):
        '''
        Returns a scalar value if the result is of length 1.
        '''
        if self.observed is None: return
        if len(self.observed) > 1:
            return str(self.observed)
        return str(self.observed[0])

    def isObserved(self):
        return self.observed is not None

    def display(self):
        '''
        Returns the kind as a string representation.
        '''
        return self.kind.show_full()


    #####################
    # Scrolling
    #####################
    def scrollDown(self):
        self.kindRow = min(self.kindRow + 1, self.rows)

    def scrollUp(self):
        self.kindRow = max(0, self.kindRow - 1)

    def scrollLeft(self):
        self.kindCol = max(0, self.kindCol - 1)
    
    def scrollRight(self):
        self.kindCol = min(self.kindCol + 1, self.cols)