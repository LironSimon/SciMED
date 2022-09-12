# library imports
import os
import json
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from gplearn.genetic import SymbolicRegressor

# project imports
from utills.plotter import Plotter
from utills.fitness_methods import *
from utills.logger_config import Logger
from utills.symbolic_regression_to_latex_text import SymbolicRegressionToLatexText


class GASF:
    """
    This class is responsible for generating a symbolic equation
    of a target value from a given set of features, using a
    SymbolicRegressor.

    The class contains 2 functions:
    1. run: Kfold trains a model and returns it fitted
    2. run_and_analyze: applies the run function multiple times
       to gain statistical insight on the performance.
    """

    # CONSTS #
    DEFAULT_TEST_FIT_FUNCTION = better_symbolic_reg_fitness
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(non_normalized_data: pd.DataFrame,
            generations: int,
            population_size: int,
            k_fold: int,
            performance_metric,
            parsimony_coefficient: float,
            verbose: int,
            expected_eq='Unknown',
            cores: int = -1):
        """
        Run the GAFS algorithm with some hyper-parameters.
        Initially the model is trained on a kfold portion of data,
        and then on the dataset as a whole.
        The model of the latter case is returned.
        """
        y_col = non_normalized_data.keys()[-1]
        x_values = non_normalized_data.drop([y_col], axis=1)
        y_values = non_normalized_data[y_col]
        # make a k-fold cross validation so we can trust the results better
        kf = KFold(n_splits=k_fold)
        scores = []
        fold_index = 1
        for train_index, test_index in kf.split(x_values):
            # say we do fold
            Logger.print(message="   Symbolic regression {} fold".format(fold_index))
            fold_index += 1
            # prepare data
            X_train, X_test = x_values.iloc[train_index, :], x_values.iloc[test_index, :]
            y_train, y_test = y_values.iloc[train_index], y_values.iloc[test_index]
            # prepare model
            est = SymbolicRegressor(population_size=population_size,
                                    generations=generations,
                                    metric=performance_metric,
                                    n_jobs=cores,
                                    verbose=verbose,
                                    parsimony_coefficient=parsimony_coefficient,
                                    random_state=73)
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            score = performance_metric(y_test, y_pred) if not isinstance(performance_metric, str) else function_mapper[performance_metric](y_test, y_pred)
            scores.append(score)

        # train a symbolic regression on all the data, it is at least as good as the previous ones
        est = SymbolicRegressor(population_size=population_size,
                                generations=generations,
                                n_jobs=cores,
                                feature_names=non_normalized_data.keys()[:-1],
                                parsimony_coefficient=parsimony_coefficient,
                                verbose=verbose,
                                random_state=73)
        est.fit(x_values, y_values)
        # if we want to compare to the EQ.
        if expected_eq != 'Unknown':
            Logger.print(message='Expected eq: {}, Found eq: {} | Found eq as latex: {}'.format(expected_eq,
                                                                                                est,
                                                                                                "NA"))#SymbolicRegressionToLatexText.run(eq=str(est))))
        else:
            Logger.print(message='Found eq: {}'.format(est))
        return est

    @staticmethod
    def run_and_analyze(run_times: int,
                        non_normalized_data: pd.DataFrame,
                        generations: int,
                        population_size: int,
                        k_fold: int,
                        performance_metric,
                        parsimony_coefficient: float,
                        save_dir: str,
                        expected_eq='Unknown',
                        cores: int = -1):
        """
        Run the GAFS algorithm several times and save results from all runs.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """
        results = pd.DataFrame()
        y_col = non_normalized_data.keys()[-1]
        x_values = non_normalized_data.drop(y_col, axis=1)
        y_values = non_normalized_data[y_col]
        current_best_wanted_loss = 99999
        best_model = None
        for test in range(run_times):
            Logger.print(message="Symbolic regression run {}".format(test + 1))
            if isinstance(parsimony_coefficient, float) and 0 <= parsimony_coefficient <= 1:
                fit_model = GASF.run(non_normalized_data=non_normalized_data,
                                     generations=generations,
                                     population_size=population_size,
                                     k_fold=k_fold,
                                     performance_metric=performance_metric,
                                     parsimony_coefficient=parsimony_coefficient,
                                     verbose=1 if test == 0 else 0,
                                     expected_eq=expected_eq,
                                     cores=cores)
            elif isinstance(parsimony_coefficient, list) and len(parsimony_coefficient) > 0 and all([isinstance(val, float) for val in parsimony_coefficient]):
                best_score = 99999
                best_inner_model = None
                best_parsimony_coefficient = 0
                score_history = {}
                for parsimony_coefficient_val in parsimony_coefficient:
                    fit_model = GASF.run(non_normalized_data=non_normalized_data,
                                         generations=generations,
                                         population_size=population_size,
                                         k_fold=k_fold,
                                         performance_metric=performance_metric,
                                         parsimony_coefficient=parsimony_coefficient_val,
                                         verbose=1 if test == 0 else 0,
                                         expected_eq=expected_eq,
                                         cores=cores)
                    try:
                        this_score = performance_metric(y_values, fit_model.predict(x_values))
                    except Exception as error:
                        this_score = GASF.DEFAULT_TEST_FIT_FUNCTION(y_values, fit_model.predict(x_values))
                    score_history[parsimony_coefficient_val] = this_score
                    if this_score > best_score:
                        best_score = this_score
                        best_inner_model = fit_model
                        best_parsimony_coefficient = parsimony_coefficient_val
                # save grid search results
                Logger.print("The best parsimony_coefficient value is: {}".format(best_parsimony_coefficient))
                with open(os.path.join(save_dir, "parsimony_coefficient_grid_search.json"), "w") as grid_search_value:
                    json.dump(score_history, grid_search_value)
                # continue with the best model
                fit_model = best_inner_model

            else:
                raise ValueError("The parsimony_coefficient argument must be either a float between 0 and 1 or a non-empty list of floats")
            pred = fit_model.predict(x_values)
            # save test scores
            try:
                wanted_loss = performance_metric(y_values, pred)
            except Exception as error:
                wanted_loss = GASF.DEFAULT_TEST_FIT_FUNCTION(y_values, pred)
            results.at[test, "wanted_loss"] = wanted_loss
            results.at[test, "mae"] = mean_absolute_error(y_values, pred)
            results.at[test, "mse"] = mean_squared_error(y_values, pred)
            results.at[test, "r2"] = r2_score(y_values, pred)
            results.at[test, "t_test_p_value"] = stats.ttest_ind(y_values, pred)[1]
            results.at[test, "found_eq"] = fit_model
            if wanted_loss < current_best_wanted_loss:
                best_model = fit_model
                current_best_wanted_loss= wanted_loss

        # print and save scoring results of all runs
        Logger.print(message="Finished all symbolic runs - ")
        [Logger.print(message="{}: {:.3} +- {:.3}".format(score, results[score].mean(), results[score].std()))
         for score in ["mae", "mse", "r2", "t_test_p_value"]]
        return results, best_model

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<GASF>"
