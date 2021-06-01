from enum import Enum


class PerformanceMeasure(Enum):
    PortfolioValue=0
    SharpeRatio=1
    EquallyWeighted=2
    Markowitz=3
    MaxmDrawDown=4
    Distance=5
    CVAR=6
    Markowitz2 = 7

class executionMode(Enum):
    train = 0
    test = 1
    validation = 2
    training_done = 3