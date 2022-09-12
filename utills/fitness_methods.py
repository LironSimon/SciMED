# library imports
import numpy as np
from sklearn.metrics import make_scorer
from gplearn.fitness import make_fitness
from sklearn.metrics import mean_absolute_error, mean_squared_error

# functions we might want to use as part of the TPOT process
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)


def simple_symbolic_reg_fitness(y: np.ndarray,
                                y_pred: np.ndarray,
                                sample_weight: np.ndarray = None) -> np.float64:
    """
    Just the MSE
    :param y_true: the list of baseline values to compare with
    :param y_pred: the list of model predicted values to evaluate
    :return: the error value from 0 to inf
    """
    if sample_weight is None:
        return mean_squared_error(y_true=y, y_pred=y_pred)
    else:
        return mean_squared_error(y_true=y, y_pred=y_pred, sample_weight=sample_weight)


def better_symbolic_reg_fitness(y: np.ndarray,
                                y_pred: np.ndarray,
                                sample_weight: np.ndarray = None) -> np.float64:
    """
    Taking ideas from https://arxiv.org/abs/1904.05417 for better overall results
    :param y_true: the list of baseline values to compare with
    :param y_pred: the list of model predicted values to evaluate
    :return: the error value from 0 to inf
    """
    if sample_weight is None:
        return mean_squared_error(y_true=y, y_pred=y_pred) + mean_absolute_error(y_true=y, y_pred=y_pred) + np.max(y-y_pred)
    else:
        return mean_squared_error(y_true=y, y_pred=y_pred, sample_weight=sample_weight) + mean_absolute_error(y_true=y, y_pred=y_pred, sample_weight=sample_weight) + np.max(y-y_pred)


# common functions
function_mapper = {
    "simple_symbolic_reg_fitness": make_fitness(function=simple_symbolic_reg_fitness, greater_is_better=False),
    "better_symbolic_reg_fitness": make_fitness(function=better_symbolic_reg_fitness, greater_is_better=False)
}
