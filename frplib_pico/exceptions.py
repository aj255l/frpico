from __future__ import annotations

class FrplibException(Exception):
    "Base exception for specialized playground user errors."
    pass

class FrplibInternalException(FrplibException):
    "Base exception for internal condtions in the library."
    pass

class MismatchedDomain(FrplibInternalException):
    "A value outside the expected domain was passed to a mapping."
    pass

class DomainDimensionError(FrplibInternalException):
    "A value outside the expected domain was passed to a mapping."
    pass

class ConstructionError(FrplibInternalException):
    "A problem was encountered creating an object."
    pass

class EvaluationError(FrplibInternalException):
    "A problem was encountered while evaluating input or derived values."
    pass

class KindError(FrplibException):
    "An error encountered in creating or using a kind."
    pass

class FrpError(FrplibException):
    "An error encountered in creating or using an FRP."
    pass

class MarketError(FrplibException):
    "An error encountered during market interactive session."
    pass

class PlaygroundError(FrplibException):
    "An error encountered during playground interactive session."
    pass

class OperationError(PlaygroundError):
    "An error encountered during a mathematical operation"
    pass

class ConditionMustBeCallable(ConstructionError):
    "A non-callable object cannot be converted to a condition."
    pass

class StatisticError(ConstructionError):
    "A problem with creating a Statistic.."
    pass

class IndexingError(PlaygroundError):
    "An error encountered during a mathematical operation"
    pass

class ComplexExpectationWarning(FrplibException):
    "Warning: Computing an expectation may be computationally inadvisable."
    pass

class NumericConversionError(FrplibException):
    "A quantity could not be converted to numeric form."
    pass
