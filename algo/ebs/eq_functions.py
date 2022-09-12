# library imports
import numpy as np
import pandas as pd

# project import


def add(a: pd.Series,
        b: pd.Series):
    return a + b


def sub(a: pd.Series,
        b: pd.Series):
    return a - b


def div(a: pd.Series,
        b: pd.Series):
    return (a / b).fillna(0).replace([np.inf, -np.inf], 0)


def mul(a: pd.Series,
        b: pd.Series):
    return a * b


FUNCTION_MAPPER = {
    add: "add",
    sub: "sub",
    mul: "mul",
    div: "div"
}

FUNCTION_LIST = [add, sub, mul, div]
