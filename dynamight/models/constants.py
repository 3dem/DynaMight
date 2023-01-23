from enum import Enum, auto


class ConsensusInitializationMode(Enum):
    EMPTY = auto()
    MAP = auto()
    MODEL = auto()


class RegularizationMode(Enum):
    EMPTY = auto()
    MAP = auto()
    MODEL = auto()
