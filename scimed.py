# library imports
import json
import pandas as pd
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


class scimed:
    """
    The main class of the project, allow other developers to load
    it and use all the SciMED pipeline at once.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(train_data_x: pd.DataFrame,
            train_data_y: pd.DataFrame,
            test_data_x: pd.DataFrame,
            test_data_y: pd.DataFrame,
            results_folder: str,
            analytical_reachment_portion: float = 0,
            numerical_run_times: int = 20,
            numerical_generations: int = 50,
            numerical_population: int = 100,
            analytical_run_times: int = 20,
            analytical_generations: int = 50,
            analytical_population: int = 100,
            parsimony_coefficient: int = 0.05,
            k_fold: int = 5,
            ebs_size_range: tuple = (5, 9),
            numerical_bool: bool = True,
            analytical_bool: bool = True,
            force_ebs_bool: bool = True,
            feature_indexes_ranges = "Not applicable",
            feature_selection_generations: int = None,
            feature_selection_pop_size: int = None,
            feature_selection_mutation_rate: float = None,
            feature_selection_royalty: float = None):
        """
        Single entry point
        """

        # 1) prepare IO
        os.makedirs(results_folder, exist_ok=True)

        # init logger
        Logger(save_path=os.path.join(results_folder, "logger.txt"))

        # 2) run the numerical part
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
                # 2.2.4) reduce the dataset of non-normalized samples for next part
                train_data_x = train_data_x.iloc[:, best_gene.feature_indexes+[-1]]
            # 2.3 add more data to the original data with the model
            # TODO: add to the next release after fixing the sample method in production

        # 3) continue to the symbolic regression
        if analytical_bool:
            # 3.1) run symbolic regression multiple times
            all_s_scores, best_s_model = GASF.run_and_analyze(run_times=analytical_run_times,
                                                              non_normalized_data=train_data_x,
                                                              performance_metric=function_mapper["better_symbolic_reg_fitness"],
                                                              generations=analytical_generations,
                                                              population_size=analytical_population,
                                                              k_fold=k_fold,
                                                              cores=-1,
                                                              parsimony_coefficient=parsimony_coefficient,
                                                              save_dir=results_folder)
            # 3.2) save results of best model from all runs
            non_norm_train_x, non_norm_test_x, non_norm_train_y, non_norm_test_y = train_test_split(train_data_x,
                                                                                                    train_data_y,
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
        else:
            ebs_flag = False

        # 4) continue to the EBS
        if ebs_flag or force_ebs_bool:
            # 4.1) run EBS multiple times
            all_ebs_scores, best_ebs_model = EBS.run_and_analyze(run_times=analytical_run_times,
                                                                 non_normalized_data=train_data_x,
                                                                 performance_metric=function_mapper[
                                                                     "better_symbolic_reg_fitness"],
                                                                 cores=-1,
                                                                 size_range=ebs_size_range,
                                                                 save_dir=results_folder)
            # 4.2) save the fitting score results
            ResultTracker.ebs_results(model=best_ebs_model,
                                      all_scores=all_ebs_scores,
                                      save_dir=results_folder)
