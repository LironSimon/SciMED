# library imports
import os
import json
import time
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split

# project imports
from utills.consts import *
from utills.fitness_methods import *
from utills.logger_config import Logger
from algo.equation_brute_search import EBS
from utills.result_tracker import ResultTracker
from algo.multi_tpot_analysis import MultiTPOTrunner
from algo.genetic_algorithm_symbolic_fit import GASF
from algo.genetic_algorithm_feature_selection import GAFS


class ExpSFF:
    """
    A father class to the SFF experiments, responsible for
    the run function for all SFF cases.
    Here, only the data generation function changes from
    case to case.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(numerical_bool: bool,
            analytical_bool: bool,
            force_ebs_bool: bool,
            results_folder: str,
            data_path: str,
            data_generation_function,
            numerical_run_times: int,
            numerical_generations: int,
            numerical_population: int,
            analytical_run_times: int,
            analytical_generations: int,
            analytical_population: int,
            parsimony_coefficient: int,
            k_fold: int,
            samples: int,
            rhoa_range: tuple,
            rhop_range: tuple,
            nu_range: tuple,
            re_range: tuple,
            ebs_size_range: tuple,
            expected_eq: str = "unknown",
            feature_selection_generations: int = None,
            feature_selection_pop_size: int = None,
            feature_selection_mutation_rate: float = None,
            feature_selection_royalty: float = None):

        # config logging
        start_time = time.time()

        # prepare IO
        os.makedirs(results_folder, exist_ok=True)
        Logger(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            results_folder,
                            "run.log"))
        # 1) generate data
        feature_indexes_ranges = data_generation_function(samples=samples,
                                                          rhoa_range=rhoa_range,
                                                          nu_range=nu_range,
                                                          re_range=re_range,
                                                          rhop_range=rhop_range,
                                                          save_path=data_path)
        # 1.1) load data, normalize and split
        df = pd.read_csv(data_path)
        Logger.print('Generated data:\n{}'.format(df.describe()))
        y_col = df.keys()[-1]
        normalized_df = (df - df.min()) / (df.max() - df.min())
        train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(normalized_df.drop([y_col], axis=1),
                                                                                normalized_df[y_col],
                                                                                shuffle=True,
                                                                                test_size=SFF_TEST_SIZE_PORTION,
                                                                                random_state=RANDOM_STATE)
        # 1.2) log elapsed time
        data_end_time = time.time()
        Logger.print("   --- Finished. Elapsed time: {} ---".format(timedelta(seconds=data_end_time - start_time)))

        # 2) continue to the MultiTPOTrunner regression
        Logger.print('Training MultiTPOTrunner:')
        if numerical_bool:
            # 2.1) run multi-tpot analysis if feature selection isn't needed
            if feature_indexes_ranges == "Not applicable":
                # 2.1.1) find the best ML model for all the data
                all_t_scores, best_t_model = MultiTPOTrunner.run_and_analyze(run_times=numerical_run_times,
                                                                             train_data_x=train_data_x,
                                                                             train_data_y=train_data_y,
                                                                             test_data_x=test_data_x,
                                                                             test_data_y=test_data_y,
                                                                             generations=numerical_generations,
                                                                             population_size=numerical_population,
                                                                             k_fold=k_fold,
                                                                             performance_metric=neg_mean_squared_error_scorer,
                                                                             save_dir=results_folder,
                                                                             n_jobs=-1)
                # 2.1.2) save results of best model from all runs
                ResultTracker.run(program_part="tpot",
                                  run_times=numerical_run_times,
                                  all_scores=all_t_scores,
                                  model=best_t_model,
                                  train_data_x=train_data_x,
                                  train_data_y=train_data_y,
                                  test_data_x=test_data_x,
                                  test_data_y=test_data_y,
                                  save_dir=results_folder)
            # 2.2) run multi-tpot analysis with feature selection
            else:
                # 2.2.1) find the best ML model for a subset of the data
                best_gene = GAFS.run(tpot_run_times=numerical_run_times,
                                     feature_generations=feature_selection_generations,
                                     tpot_regressor_generations=numerical_generations,
                                     feature_population_size=feature_selection_pop_size,
                                     tpot_regressor_population_size=numerical_population,
                                     mutation_rate=feature_selection_mutation_rate,
                                     feature_indexes_ranges=feature_indexes_ranges,
                                     mutation_w=[val[1]-val[0] for val in feature_indexes_ranges],
                                     royalty=feature_selection_royalty,
                                     k_fold=k_fold,
                                     performance_metric=neg_mean_squared_error_scorer,
                                     train_data_x=train_data_x,
                                     train_data_y=train_data_y,
                                     test_data_x=test_data_x,
                                     test_data_y=test_data_y,
                                     save_dir=results_folder,
                                     cores=-1)
                # 2.2.2) save results of best model from all runs
                ResultTracker.run(program_part="tpot",
                                  run_times=numerical_run_times,
                                  all_scores=best_gene.scoring_history,
                                  model=best_gene.model_object,
                                  train_data_x=train_data_x.iloc[:, best_gene.feature_indexes],
                                  train_data_y=train_data_y,
                                  test_data_x=test_data_x.iloc[:, best_gene.feature_indexes],
                                  test_data_y=test_data_y,
                                  save_dir=results_folder)
                # 2.2.3) save selected features of best gene
                with open(os.path.join(os.path.dirname(__file__), results_folder, "best_features_selected.json"),
                          "w") as features_file:
                    json.dump({"index": best_gene.feature_indexes,
                               "names": list(test_data_x.columns[best_gene.feature_indexes])},
                              features_file)
                Logger.print("Best gene features: {}".format(list(test_data_x.columns[best_gene.feature_indexes])))
                # 2.2.4) reduce the dataset of non-normalized samples for next part
                df = df.iloc[:, best_gene.feature_indexes+[-1]]
        # 2.3) log elapsed time
        tpot_end_time = time.time()
        symbolic_end_time = time.time()
        Logger.print("   --- Finished. Elapsed time: {} ---".format(timedelta(seconds=tpot_end_time-data_end_time)))

        # 3) continue to the symbolic regression
        if analytical_bool:
            Logger.print('Searching for a symbolic expression:')
            # 3.1) run symbolic regressor multiple times
            all_s_scores, best_s_model = GASF.run_and_analyze(run_times=analytical_run_times,
                                                              non_normalized_data=df,
                                                              performance_metric=function_mapper["better_symbolic_reg_fitness"],
                                                              generations=analytical_generations,
                                                              population_size=analytical_population,
                                                              k_fold=k_fold,
                                                              cores=-1,
                                                              parsimony_coefficient=parsimony_coefficient,
                                                              expected_eq=expected_eq,
                                                              save_dir=results_folder)
            # 3.2) save results of best model from all runs
            non_norm_train_x, non_norm_test_x, non_norm_train_y, non_norm_test_y = train_test_split(df.drop([y_col], axis=1),
                                                                                                    df[y_col],
                                                                                                    shuffle=True,
                                                                                                    test_size=SFF_TEST_SIZE_PORTION,
                                                                                                    random_state=RANDOM_STATE)
            p_value_flag = ResultTracker.run(program_part="symbolic",
                                             run_times=analytical_run_times,
                                             all_scores=all_s_scores,
                                             model=best_s_model,
                                             train_data_x=non_norm_train_x,
                                             train_data_y=non_norm_train_y,
                                             test_data_x=non_norm_test_x,
                                             test_data_y=non_norm_test_y,
                                             save_dir=results_folder)
            # 3.3) save a summary of the eqs found & figure whether to continue to ebf
            stability_flag = ResultTracker.summaries_symbolic_results(run_times=analytical_run_times,
                                                                percent_of_majority=SYMBOLIC_PERCENT_OF_MAJORITY,
                                                                eq_ranking_metric=SYMBOLIC_EQ_RANKING_METRIC,
                                                                top_eqs_max_num=SYMBOLIC_TOP_EQS_MAX_NUM,
                                                                save_dir=results_folder)

            ebs_flag = p_value_flag or stability_flag
            # 3.4) log elapsed time
            symbolic_end_time = time.time()
            Logger.print("Finished. Elapsed time: {}".format(timedelta(seconds=tpot_end_time - tpot_end_time)))
        else:
            ebs_flag = False

        # 4) continue to the EBS
        if ebs_flag or force_ebs_bool:
            Logger.print('Searching for a symbolic expression using EBF:')
            # 4.1) run EBS multiple times
            all_ebs_scores, best_ebs_model = EBS.run_and_analyze(run_times=analytical_run_times,
                                                                 non_normalized_data=df,
                                                                 performance_metric=function_mapper[
                                                                     "better_symbolic_reg_fitness"],
                                                                 cores=-1,
                                                                 size_range=ebs_size_range,
                                                                 expected_eq=expected_eq,
                                                                 save_dir=results_folder)
            # 4.2) save the fitting score results
            ResultTracker.ebs_results(model=best_ebs_model,
                                      all_scores=all_ebs_scores,
                                      save_dir=results_folder)
        else:
            Logger.print("EBF search of a symbolic equation wasn't needed")
            # 4.3) log elapsed time
            ebs_end_time = time.time()
            Logger.print("Finished. Elapsed time: {}".format(timedelta(seconds=ebs_end_time - symbolic_end_time)))

        # 5) alert results to the user
        Logger.print("\n   --- TOTAL TIME ELAPSED TIME: {} ---".format(timedelta(seconds=time.time() - start_time)))
