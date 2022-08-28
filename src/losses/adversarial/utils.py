from enum import Enum


class AdversarialLosses(Enum):
    VANILLA = "vanilla"
    HINGE = "hinge"
    LEAST_SQUARE = "least_square"
