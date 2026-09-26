from typing import NamedTuple
from pandas import Index 
from numpy import ndarray

class RegressionResults(NamedTuple):
    signs_column: Index
    k: ndarray
    b: float
    mae: float
    mse: float
    r2: float
    y_predict: float
    y_real: float

class ClassiferResults(NamedTuple):
    importances: ndarray
    accuracy: float
    report: dict
    matrix:ndarray
