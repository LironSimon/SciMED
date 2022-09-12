# library imports
import os
import json
import pandas as pd
from scipy import stats
import concurrent.futures
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# project imports
from algo.ebs.eq import Eq
from utills.plotter import Plotter
from algo.ebs.eq_node import EqNode
from algo.ebs.eq_functions import *
from utills.fitness_methods import *
from utills.logger_config import Logger


class EBS:
    """
    This class is responsible for generating a symbolic equation
    of a target value from a given set of features, using a tree-structure brute-search.

    The class contains 2 functions:
    1. run: Kfold trains a model and returns it fitted
    2. run_and_analyze: applies the run function multiple times
       to gain statistical insight on the performance.
    """

    # CONSTS #
    DEFAULT_TEST_FIT_FUNCTION = better_symbolic_reg_fitness
    # END - CONSTS #

    # CATCH FOR FASTER COMPUTATION #
    TOPOLOGY_TREES = {}
    ALLOCATED_EQS = {}
    # END - CATCH FOR FASTER COMPUTATION #

    def __init__(self):
        pass

    @staticmethod
    def run(non_normalized_data: pd.DataFrame,
            k_fold: int,
            performance_metric,
            verbose: int,
            size_range: tuple,
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
            Logger.print(message="Equation brute force {} fold".format(fold_index))
            fold_index += 1
            # prepare data
            X_train, X_test = x_values.iloc[train_index, :], x_values.iloc[test_index, :]
            y_train, y_test = y_values.iloc[train_index], y_values.iloc[test_index]
            # prepare model
            eq, best_score, answer = EBS._search_equation(x=X_train,
                                                          y=y_train,
                                                          performance_metric=performance_metric,
                                                          verbose=verbose,
                                                          cores=cores,
                                                          size_range=size_range)
            y_pred = eq.eval(X_test)
            score = performance_metric(y_test, y_pred) if not isinstance(performance_metric, str) else function_mapper[
                performance_metric](y_test, y_pred)
            scores.append(score)

        # train a symbolic regression on all the data, it is at least as good as the previous ones
        eq, best_score, answer = EBS._search_equation(x=x_values,
                                                      y=y_values,
                                                      performance_metric=performance_metric,
                                                      verbose=verbose,
                                                      cores=cores,
                                                      size_range=size_range)
        # if we want to compare to the EQ.
        if expected_eq != 'Unknown':
            Logger.print(message='Expected eq: {}, Found eq: {}'.format(expected_eq,
                                                                        eq.to_string()))
        else:
            Logger.print(message='Found eq: {}'.format(eq.to_string()))
        return eq

    @staticmethod
    def run_and_analyze(run_times: int,
                        non_normalized_data: pd.DataFrame,
                        performance_metric,
                        save_dir: str,
                        size_range: tuple,
                        expected_eq='Unknown',
                        cores: int = -1):
        """
        Run the GAFS algorithm several times and save results from all runs.
        Returns a pandas dataframe of all results and the best model from
        all runs.

        @:var size_range - start, end, and step size of the EQ tree's number of nodes
        """
        results = pd.DataFrame()
        y_col = non_normalized_data.keys()[-1]
        x_values = non_normalized_data.drop(y_col, axis=1)
        y_values = non_normalized_data[y_col]
        current_best_wanted_loss = 99999
        best_model = None
        for test in range(run_times):
            Logger.print(message="run {}".format(test + 1))
            eq, best_score, answer = EBS._search_equation(x=x_values,
                                                          y=y_values,
                                                          verbose=1 if test == 0 else 0,
                                                          performance_metric=performance_metric,
                                                          size_range=size_range,
                                                          cores=cores)
            pred = eq.eval(x_values)
            # save test scores
            try:
                wanted_loss = performance_metric(y_values, pred)
            except Exception as error:
                wanted_loss = EBS.DEFAULT_TEST_FIT_FUNCTION(y_values, pred)
            results.at[test, "wanted_loss"] = wanted_loss
            results.at[test, "mae"] = mean_absolute_error(y_values, pred)
            results.at[test, "mse"] = mean_squared_error(y_values, pred)
            results.at[test, "r2"] = r2_score(y_values, pred)
            results.at[test, "t_test_p_value"] = stats.ttest_ind(y_values, pred)[1]
            results.at[test, "found_eq"] = eq.to_string()
            if wanted_loss < current_best_wanted_loss or best_model is None:
                best_model = eq
                current_best_wanted_loss = wanted_loss

        # print and save scoring results of all runs
        Logger.print(message="Finished all EBS runs - ")
        if expected_eq != 'Unknown':
            Logger.print(message='Expected eq: {}, Found eq: {}'.format(expected_eq,
                                                                        best_model.to_string()))
        else:
            Logger.print(message='Found eq: {}'.format(best_model.to_string()))
        [Logger.print(message="{}: {:.3} +- {:.3}".format(score, results[score].mean(), results[score].std()))
         for score in ["mae", "mse", "r2", "t_test_p_value"]]
        results.to_csv(os.path.join(save_dir, "ebs_scoring_history.csv"))
        # plot best model's predictions vs true values
        Plotter.y_test_vs_y_pred(model=best_model,
                                 x_test=x_values,
                                 y_test=y_values,
                                 save_path=os.path.join(save_dir, "ebs_target_vs_pred.pdf"))
        return results, best_model

    @staticmethod
    def _search_equation(x: pd.DataFrame,
                         y: pd.Series,
                         verbose: int,
                         performance_metric,
                         size_range: tuple,
                         cores: int) -> tuple:
        """
        Search for the equation
        # TODO: think how to use multi-thread later
        """
        # run over the needed range to generate all possible tree topologies
        for n in size_range:
            if verbose == 1:
                Logger.print(message="EBS._search_equation: Generating all possible binary tree topologies for size {}".format(n))

            if n < -1:
                continue
            elif (n % 2) == 0:
                EBS.TOPOLOGY_TREES[n] = []
            elif n == 1:
                EBS.TOPOLOGY_TREES[1] = [Eq(tree=EqNode(value=None))]
            elif n not in EBS.TOPOLOGY_TREES:  # do not calc the same topology twice
                EBS.TOPOLOGY_TREES[n] = EBS._all_possible_fbt(n=n)
        # find best equation for the data
        answer = {}
        best_eq = ""
        best_score = 9999
        # run over the tree topologies sizes
        for n in size_range:
            if verbose == 1:
                Logger.print(message="EBS._search_equation: Testing {} possible binary tree topologies for size {}".format(n, len(EBS.TOPOLOGY_TREES[n])))

            # run over all tree populations
            for tree_topology in EBS.TOPOLOGY_TREES[n]:
                tree_topology.fix_nodes()  # just to make sure the meta-values are fine
                # avoid computing the same data twice, if we have this allocation list, use it and calc if we don't
                if tree_topology.to_id_str() not in EBS.ALLOCATED_EQS:
                    # populate each tree with all possible combinations
                    possible_trees = EBS._populate_tree(eq=tree_topology,
                                                        not_leaf_values=FUNCTION_LIST,
                                                        leaf_values=list(x))
                    EBS.ALLOCATED_EQS[tree_topology.to_id_str()] = possible_trees  # recall the allocation list
                else:
                    possible_trees = EBS.ALLOCATED_EQS[tree_topology.to_id_str()]

                if verbose == 1:
                    Logger.print(message="EBS._search_equation: Found {} possible populated trees to check for this topology".format(len(possible_trees)))
                # for each combination compute performance
                for eq_index, this_eq in enumerate(possible_trees):
                    # TODO: change this magic number later
                    if (eq_index % 100) == 0 and verbose == 1:
                        Logger.print(message="EBS._search_equation: Test of {} / {} ({:.3f}%) equations done".format(eq_index, len(possible_trees), eq_index*100/len(possible_trees)))
                    try:
                        y_pred = this_eq.eval(x_values=x)
                        reg = LinearRegression().fit([[val] for val in y_pred], y)
                        this_eq.linear_a = reg.coef_[0]
                        this_eq.linear_b = reg.intercept_
                        y_pred = this_eq.linear_a * y_pred + this_eq.linear_b  # re-calibrate results
                        score = performance_metric(y, y_pred)  # calc the performance
                        answer[this_eq.to_string()] = score
                        if score < best_score:
                            best_score = score
                            best_eq = this_eq
                    except Exception as error:
                        this_eq.eval(x_values=x)
                        Logger.debug(message="Error at EBS._search_equation, saying: {}".format(error))
        return best_eq, best_score, answer

    @staticmethod
    def _all_possible_fbt(n: int) -> list:
        return Eq.all_possible_fbt(n=n)

    @staticmethod
    def _populate_tree(eq: Eq,
                       not_leaf_values: list,
                       leaf_values: list) -> list:
        """
        Gets a tree topology and return all possible value allocations to the tree
        :param tree: the tree topology
        :param not_leaf_values: the function can be allocated to the not leaf node
        :param leaf_values: the leaves values
        :return: all possible allocation of the values to the given
        """
        return eq.populate(not_leaf_values=not_leaf_values,
                           leaf_values=leaf_values)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<EBS>"
