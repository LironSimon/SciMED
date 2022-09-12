# library imports
import os
import pickle
import pandas as pd
from scipy import stats
from tpot import TPOTRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# project imports
from utills.logger_config import Logger
from utills.tpot_results_extractor import TPOTresultsExtractor


class MultiTPOTrunner:
    """
    This class is responsible for generating a numerical
    prediction of a target value from a given set of features,
    using a TPOTRegressor.

    Input dataset is used to train and test the model
    multiple times (named as run_times), to gain statistical
    insight on the performance.
    """

    def __init__(self):
        pass

    @staticmethod
    def run_and_analyze(run_times: int,
                        train_data_x: pd.DataFrame,
                        train_data_y: pd.DataFrame,
                        test_data_x: pd.DataFrame,
                        test_data_y: pd.DataFrame,
                        generations: int,
                        population_size: int,
                        k_fold: int,
                        performance_metric,
                        save_dir: str,
                        n_jobs: int = -1):
        """
        Run the TPOTRegressor algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """
        # const
        tpot_model_file_path = os.path.join(save_dir, 'current_tpot_pipeline.py')
        tpot_object_file_path = os.path.join(save_dir, 'current_tpot_pipeline_as_object')
        # prepare DF for results
        best_model = None
        results = pd.DataFrame()
        current_best_performance_score = 99999
        for test in range(run_times):
            Logger.print("TPOT run {}/{}".format(test + 1, run_times))
            model = TPOTRegressor(generations=generations,
                                  population_size=population_size,
                                  cv=KFold(n_splits=k_fold),
                                  scoring=performance_metric,
                                  verbosity=2,
                                  n_jobs=n_jobs)
            model.fit(train_data_x, train_data_y)
            pred = model.predict(test_data_x)
            # store test scores
            try:
                # we assume this is just a function
                performance_score = performance_metric(test_data_y, pred)
            except:
                # maybe it is a scorer wrapper of a function and we want to overcome it
                performance_score = performance_metric._score_func(test_data_y, pred)
            results.at[test, "performance_score"] = performance_score
            results.at[test, "mae"] = mean_absolute_error(test_data_y, pred)
            results.at[test, "mse"] = mean_squared_error(test_data_y, pred)
            results.at[test, "r2"] = r2_score(test_data_y, pred)
            results.at[test, "t_test_p_value"] = stats.ttest_ind(test_data_y, pred)[1]
            # store exported pipeline
            model.export(tpot_model_file_path)
            pipeline = TPOTresultsExtractor.process_file(tpot_model_file_path)
            results.at[test, 'pipeline'] = pipeline
            # update best mae score and model
            if performance_score < current_best_performance_score:
                best_model = model
                current_best_performance_score = performance_score
        # remove unnecessary file
        os.remove(tpot_model_file_path)
        # Logger.print and save scoring results of all runs
        Logger.print("\nFinished all MultiTPOT runner runs")
        [Logger.print("{}: {:.3}+-{:.3}".format(score, results[score].mean(), results[score].std())) for score in results.keys() if score != "pipeline"]
        return results, best_model

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<MultiTPOTrunner>"
