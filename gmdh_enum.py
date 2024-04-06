"""
Matrix split type
"""


class DataMatrixSplitType:
    HALF_AND_HALF_SPLIT = 0
    """Split the matrix into two equal parts"""

    PAIRED_UNPAIRED_SPLIT = 1
    """Split the matrix into paired and unpaired sets"""

    PERCENTAGE_SPLIT = 2
    """Split the matrix based on percentage"""


"""
Deviation kind
"""


class DeviationKind:
    ABS_DEVIATION = 0
    """Absolute deviation"""

    REL_DEVIATION = 1
    """Relative deviation"""

    MSE_DEVIATION = 2
    """Mean squared error deviation"""

    ABS_DEVIATION_QUEUE = 3
    """Absolute deviation in a queue"""

    REL_DEVIATION_QUEUE = 4
    """Relative deviation in a queue"""

    MSE_DEVIATION_QUEUE = 5
    """Mean squared error deviation in a queue"""

    ABS_DEVIATION_BEST_MODEL = 6
    """Absolute deviation in the best model"""

    REL_DEVIATION_BEST_MODEL = 7
    """Relative deviation in the best model"""

    MSE_DEVIATION_BEST_MODEL = 8
    """Mean squared error deviation in the best model"""


"""
GMDH polynomial kind
"""


class GMDHModelKind:
    LINE_TWO_UNKNOWN_POLYNOMIAL = 0
    """y = a + b * x1 + c * x2"""

    FIRST_DEGREE_TWO_UNKNOWN_POLYNOMIAL = 1
    """y = a + b * x1 + c * x2 + d * x1 * x2"""

    SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL = 2
    """y = a + b * x1 + c * x2 + d * x1 * x1 + e * x2 * x2 + f * x1 * x2"""

    COMPLEX_SECOND_DEGREE_TWO_UNKNOWN_POLYNOMIAL = 3
    """y = a + b * x1 + c * x2 + d * x1 * x2 + e * x2 * x1 * x1 + f * x1 * x2 * x2 + g * x1 * x1 * x2 * x2"""

